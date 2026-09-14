import unittest
from copy import deepcopy
from types import SimpleNamespace

from my_ReAct.base_tool import BaseTool
from my_ReAct.context_manager import BudgetContextManager, ContextManager, RecentTurnsContextManager
from my_ReAct.openai_llm import OpenAIResponsesLLM
from my_ReAct.react import LLMResult, ReActAgent, ToolCall
from my_ReAct.run_state import RunState
from my_ReAct.toolset import ResearchTools


class DummyTool(BaseTool):
    def __init__(self, name):
        super().__init__(name, name, {'type': 'object', 'properties': {}})
        self.closed = False

    def _execute(self):
        return {'text': 'full evidence'}

    def close(self):
        self.closed = True


class ContextTest(unittest.TestCase):
    def make_agent(self, llm, manager=None, steps=2):
        return ReActAgent(llm, steps, 'question', ResearchTools(
            *(DummyTool(name) for name in ('search', 'visit', 'python'))
        ), context_manager=manager)

    def test_window_preserves_task_and_parallel_call_pairs(self):
        prefix = [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'question'}]
        blocks = []
        for i in range(3):
            blocks += [{'role': 'assistant', 'content': None, 'tool_calls': [
                {'call_id': f'{i}-a'}, {'call_id': f'{i}-b'},
            ]}, {'role': 'tool', 'call_id': f'{i}-a', 'content': {}},
                {'role': 'tool', 'call_id': f'{i}-b', 'content': {}}]
        final = [{'role': 'user', 'content': 'answer now'}]
        messages = prefix + blocks + final
        before = deepcopy(messages)
        view = RecentTurnsContextManager(1).process(messages, RunState())
        self.assertEqual(view[:2], prefix)
        self.assertEqual(view[-4:], blocks[-3:] + final)
        self.assertEqual(messages, before)

    def test_manager_runs_on_final_call_without_mutating_trajectory(self):
        class Manager(ContextManager):
            def reset(self):
                self.steps = []

            def process(self, messages, state):
                self.steps.append(state.step)
                for message in messages:
                    if message['role'] == 'tool':
                        message['content']['data']['text'] = 'short evidence'
                return messages

        class LLM:
            def generate(self, messages, tools=None, state=None):
                if state is not None:
                    raise AssertionError('Managed history must not use continuation')
                if tools:
                    return LLMResult(tool_calls=[ToolCall('search', {}, 'c')], continuation={'id': 'old'})
                if messages[-2]['content']['data']['text'] != 'short evidence':
                    raise AssertionError('Final call missed context manager')
                return LLMResult(text='answer')

        manager = Manager()
        agent = self.make_agent(LLM(), manager, steps=1)
        result = agent.run()
        self.assertEqual(result.output, 'answer')
        self.assertEqual(manager.steps, [0, 1])
        self.assertEqual(agent.state.step, 2)
        self.assertEqual(agent.state.tool_rounds, 1)
        self.assertEqual(agent.state.stop_reason, 'tool_budget_exhausted')
        self.assertEqual(agent.messages[3]['content']['data']['text'], 'full evidence')
        self.assertTrue(all(tool.closed for tool in agent.tools.as_list()))
        self.assertEqual(agent.state.elapsed, agent.state.elapsed)
        with self.assertRaises(RuntimeError):
            agent.run()

    def test_manager_failure_closes_tools(self):
        class Broken(ContextManager):
            def process(self, messages, state):
                raise ValueError('context failed')
        agent = self.make_agent(None, Broken())
        self.assertEqual(agent.run().status, 'error')
        self.assertEqual(agent.state.step, 0)
        self.assertEqual(agent.state.stop_reason, 'runtime_error')
        self.assertTrue(all(tool.closed for tool in agent.tools.as_list()))

    def test_budget_manager_compacts_observations_and_keeps_pairs(self):
        messages = [
            {'role': 'system', 'content': 'rules'},
            {'role': 'user', 'content': 'question'},
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'name': 'visit', 'arguments': {'reference': 'doc-1'}, 'call_id': 'c1'},
            ]},
            {'role': 'tool', 'name': 'visit', 'call_id': 'c1', 'content': {
                'ok': True,
                'data': {
                    'document': {'content': 'x' * 20_000},
                    'extraction': {'status': 'ok', 'content': 'e' * 20_000},
                },
            }},
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'name': 'search', 'arguments': {'queries': []}, 'call_id': 'c2'},
            ]},
            {'role': 'tool', 'name': 'search', 'call_id': 'c2', 'content': {
                'ok': True,
                'data': {'results': [{'chunks': [
                    {'doc_id': 'doc-2', 'snippet': 's' * 2_000},
                ]}]},
            }},
        ]
        before = deepcopy(messages)
        manager = BudgetContextManager(max_chars=3_000)
        view = manager.process(messages, RunState())

        self.assertLessEqual(len(__import__('json').dumps(view, ensure_ascii=False)), 3_000)
        self.assertEqual(messages, before)
        self.assertEqual('assistant', view[-2]['role'])
        self.assertEqual('tool', view[-1]['role'])
        self.assertLess(len(view[-1]['content']['data']['results'][0]['chunks'][0]['snippet']), 2_000)
        self.assertEqual(1, manager.stats['calls'])
        self.assertTrue(manager.stats['last_truncated'])
        self.assertGreaterEqual(manager.stats['last_omitted_rounds'], 1)

    def test_responses_replays_calls_and_outputs_without_continuation(self):
        requests = []
        def create(**kwargs):
            requests.append(kwargs)
            return SimpleNamespace(id='new', output=[], output_text='answer', usage=None)
        adapter = OpenAIResponsesLLM('model', 'key')
        adapter.client = SimpleNamespace(responses=SimpleNamespace(create=create))
        adapter.generate([
            {'role': 'user', 'content': 'question'},
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'call_id': 'c', 'name': 'search', 'arguments': {}}]},
            {'role': 'tool', 'call_id': 'c', 'content': {'ok': True}},
        ])
        items = requests[0]['input']
        self.assertEqual(items[1]['type'], 'function_call')
        self.assertEqual(items[2]['type'], 'function_call_output')
        self.assertEqual(items[1]['call_id'], items[2]['call_id'])
        self.assertNotIn('previous_response_id', requests[0])


if __name__ == '__main__':
    unittest.main()
