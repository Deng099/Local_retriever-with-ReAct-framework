import json
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field

from my_ReAct.base_tool import RegisterTool
from my_ReAct.contracts import AgentRunResult
from my_ReAct.prompts import build_agent_system_prompt
from my_ReAct.toolset import ResearchTools
from my_ReAct.run_state import RunState


@dataclass
class ToolCall:
    name: str
    arguments: dict
    call_id: str | None = None


@dataclass
class LLMResult:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    continuation: object | None = None
    usage: object | None = None


class Base_LLM:
    def generate(self, messages, tools=None, state=None):
        # 要求所有子类override，否则raise error
        raise NotImplementedError


class ParseBasedLLM(Base_LLM):
    '''对于没有原生 tool calling API 的模型，要求模型生成 <tool_call>...</tool_call>，然后统一转换成 LLMResult'''
    def generate_text(self, messages, tool=None):
        raise NotImplementedError

    def generate(self, messages, tools=None, state=None):
        response = self.generate_text(messages, tools)
        tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL)
        if not tool_call_match:
            return LLMResult(text=response)

        tool_call = json.loads(tool_call_match.group(1).strip())
        return LLMResult(text=response, tool_calls=[ToolCall(tool_call['name'], tool_call['arguments'])])


class ReActAgent:
    def __init__(self, llm, max_step, query, tools: ResearchTools, context=None, context_manager=None):
        self.llm = llm
        self.max_step = max_step
        self.context = context
        self.context_manager = context_manager
        self.state = RunState()
        self._has_run = False
        self.tools = tools
        self.tool_registry = RegisterTool(tools.as_list())
        self.tool_schemas = self.tool_registry.get_tool_schemas()
        self.llm_state = None
        system_prompt = build_agent_system_prompt(self.tool_schemas)
        self.messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': query},
        ]

    @property
    def llm_state(self):
        return self.state.continuation

    @llm_state.setter
    def llm_state(self, value):
        self.state.continuation = value

    def _generate(self, tools):
        messages = deepcopy(self.messages)
        if self.context_manager is not None:
            messages = self.context_manager.process(messages, self.state)
            # A managed view is authoritative. Do not reuse hidden provider
            # history, even on later calls where the view has the same size.
            self.llm_state = None
        result = self.llm.generate(messages, tools, state=self.llm_state)
        self.state.step += 1
        self.llm_state = result.continuation
        return result

    def run(self):
        # Tools (including the Python session) are closed at the end of a run.
        if self._has_run:
            raise RuntimeError('Create a new Agent for each run; its tools have a single-run lifecycle')
        self._has_run = True
        self.state = RunState()
        result = None
        try:
            if self.context_manager is not None:
                self.context_manager.reset()
            result = self._run_loop()
        except Exception as exc:
            result = AgentRunResult.failed(type(exc).__name__, str(exc))
        finally:
            close_errors = self.tool_registry.close_all()
        if close_errors and result.status != 'error':
            first_error = close_errors[0]
            result = AgentRunResult.failed(
                first_error['type'],
                f"Failed to close tool '{first_error['tool']}': {first_error['message']}",
                termination_reason='resource_cleanup_error',
            )
        self.state.finish(result.termination_reason)
        return result

    def _run_loop(self):
        for step in range(self.max_step):

            llm_started = time.perf_counter()
            llm_result = self._generate(self.tool_schemas)
            assistant_message = {
                'role': 'assistant',
                'content': llm_result.text,
                'latency_seconds': time.perf_counter() - llm_started,
            }
            if llm_result.tool_calls:
                assistant_message['tool_calls'] = [
                    {'name': call.name, 'arguments': call.arguments, 'call_id': call.call_id}
                    for call in llm_result.tool_calls
                ]
            if llm_result.usage is not None:
                assistant_message['usage'] = llm_result.usage.model_dump() if hasattr(llm_result.usage, 'model_dump') else llm_result.usage
            self.messages.append(assistant_message)

            if not llm_result.tool_calls:
                if llm_result.text:
                    return AgentRunResult.completed(llm_result.text)
                return AgentRunResult.incomplete('empty_answer')

            self.state.tool_rounds += 1
            for tool_call in llm_result.tool_calls:
                # 从llm得到name，arguments
                name = tool_call.name
                parameters = tool_call.arguments
                
                # < >name, parameter, < >
                tool_started = time.perf_counter()
                tool_result = self.tool_registry.execute(name, **parameters)

                self.messages.append({
                    'role': 'tool',
                    'name': name,
                    'content': tool_result,
                    'call_id': tool_call.call_id,
                    'latency_seconds': time.perf_counter() - tool_started,
                })

        self.messages.append({
            'role': 'user',
            'content': 'The tool budget is exhausted. Do not call tools. Answer the original question now using the available evidence.'
        })
        llm_started = time.perf_counter()
        final_result = self._generate([])
        final_message = {
            'role': 'assistant',
            'content': final_result.text,
            'latency_seconds': time.perf_counter() - llm_started,
        }
        if final_result.usage is not None:
            final_message['usage'] = final_result.usage.model_dump() if hasattr(final_result.usage, 'model_dump') else final_result.usage
        self.messages.append(final_message)
        if final_result.tool_calls:
            return AgentRunResult.incomplete('unexpected_tool_call_after_budget', final_result.text)
        if final_result.text:
            return AgentRunResult.completed(final_result.text, 'tool_budget_exhausted')
        return AgentRunResult.incomplete('tool_budget_exhausted_empty')
