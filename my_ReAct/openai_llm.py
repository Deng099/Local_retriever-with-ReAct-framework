import json

from openai import DefaultHttpxClient, OpenAI

from my_ReAct.react import Base_LLM, LLMResult, ToolCall


class OpenAICompatibleChatLLM(Base_LLM):
    '''使用 OpenAI-compatible Chat Completions 原生 tool calling，并转换成统一的 LLMResult'''
    def __init__(self, model, api_key, base_url=None, trust_env=True):
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=DefaultHttpxClient(trust_env=trust_env),
        )

    def generate(self, messages, tools=None, state=None):
        request = {'model': self.model, 'messages': self._provider_messages(messages)}
        if tools:
            request['tools'] = tools
            request['tool_choice'] = 'auto'
        response = self.client.chat.completions.create(**request)
        message = response.choices[0].message
        tool_calls = [
            ToolCall(call.function.name, json.loads(call.function.arguments), call.id)
            for call in message.tool_calls or []
        ]
        return LLMResult(text=message.content or None, tool_calls=tool_calls, usage=response.usage)

    @staticmethod
    def _provider_messages(messages):
        provider_messages = []
        for message in messages:
            role = message['role']
            if role in {'system', 'user'}:
                provider_messages.append({'role': role, 'content': message['content']})
            elif role == 'assistant':
                converted = {'role': 'assistant', 'content': message.get('content')}
                if message.get('tool_calls'):
                    converted['tool_calls'] = [
                        {
                            'id': call['call_id'],
                            'type': 'function',
                            'function': {
                                'name': call['name'],
                                'arguments': json.dumps(call['arguments'], ensure_ascii=False),
                            },
                        }
                        for call in message['tool_calls']
                    ]
                provider_messages.append(converted)
            elif role == 'tool':
                provider_messages.append({
                    'role': 'tool',
                    'tool_call_id': message['call_id'],
                    'content': json.dumps(message['content'], ensure_ascii=False),
                })
        return provider_messages


class OpenAIResponsesLLM(Base_LLM):
    '''使用 OpenAI Responses API 原生 tool calling，并转换成统一的 LLMResult'''
    def __init__(self, model, api_key, base_url=None, reasoning_effort=None, trust_env=True):
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=DefaultHttpxClient(trust_env=trust_env),
        )
        self.reasoning_effort = reasoning_effort

    def generate(self, messages, tools=None, state=None):
        provider_tools = [{'type': 'function', **schema['function']} for schema in tools or []]
        if state:
            pending_call_ids = set(state.get('pending_call_ids', []))
            input_items = [
                {
                    'type': 'function_call_output',
                    'call_id': message['call_id'],
                    'output': json.dumps(message['content'], ensure_ascii=False),
                }
                for message in messages
                if message['role'] == 'tool' and message['call_id'] in pending_call_ids
            ]
            if not tools and messages[-1]['role'] == 'user':
                input_items.append(messages[-1])
        else:
            input_items = self._replay_messages(messages)

        request = {'model': self.model, 'input': input_items}
        if provider_tools:
            request['tools'] = provider_tools
        if state:
            request['previous_response_id'] = state['previous_response_id']
        if self.reasoning_effort:
            request['reasoning'] = {'effort': self.reasoning_effort}

        response = self.client.responses.create(**request)
        tool_calls = [
            ToolCall(item.name, json.loads(item.arguments), item.call_id)
            for item in response.output
            if item.type == 'function_call'
        ]
        return LLMResult(
            text=response.output_text or None,
            tool_calls=tool_calls,
            continuation={
                'previous_response_id': response.id,
                'pending_call_ids': [tool_call.call_id for tool_call in tool_calls],
            },
            usage=response.usage,
        )

    @staticmethod
    def _replay_messages(messages):
        """Rebuild visible history when no provider continuation is used."""
        items = []
        for message in messages:
            role = message['role']
            if role in {'system', 'user'}:
                items.append({'role': role, 'content': message['content']})
            elif role == 'assistant':
                if message.get('content'):
                    items.append({'role': role, 'content': message['content']})
                for call in message.get('tool_calls', []):
                    items.append({
                        'type': 'function_call', 'call_id': call['call_id'],
                        'name': call['name'],
                        'arguments': json.dumps(call['arguments'], ensure_ascii=False),
                    })
            elif role == 'tool':
                items.append({
                    'type': 'function_call_output', 'call_id': message['call_id'],
                    'output': json.dumps(message['content'], ensure_ascii=False),
                })
        return items
