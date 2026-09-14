import json
import unittest
from types import SimpleNamespace

from my_ReAct.openai_llm import OpenAICompatibleChatLLM, OpenAIResponsesLLM


class _Recorder:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return next(self.responses)


def _chat_response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=[]))],
        usage=None,
    )


class LLMAdapterTest(unittest.TestCase):
    def test_chat_adapter_does_not_keep_hidden_history_between_runs(self):
        recorder = _Recorder([_chat_response('first answer'), _chat_response('second answer')])
        adapter = OpenAICompatibleChatLLM('model', 'key', 'http://example.test/v1')
        adapter.client = SimpleNamespace(chat=SimpleNamespace(completions=recorder))

        adapter.generate([
            {'role': 'system', 'content': 'system'},
            {'role': 'user', 'content': 'first question'},
        ])
        adapter.generate([
            {'role': 'system', 'content': 'system'},
            {'role': 'user', 'content': 'second question'},
        ])

        self.assertEqual('first question', recorder.requests[0]['messages'][-1]['content'])
        self.assertEqual('second question', recorder.requests[1]['messages'][-1]['content'])
        self.assertNotIn('first question', json.dumps(recorder.requests[1]))

    def test_chat_adapter_reconstructs_tool_messages_from_trajectory(self):
        recorder = _Recorder([_chat_response('answer')])
        adapter = OpenAICompatibleChatLLM('model', 'key', 'http://example.test/v1')
        adapter.client = SimpleNamespace(chat=SimpleNamespace(completions=recorder))
        messages = [
            {'role': 'system', 'content': 'system'},
            {'role': 'user', 'content': 'question'},
            {
                'role': 'assistant',
                'content': None,
                'tool_calls': [{
                    'name': 'search',
                    'arguments': {'queries': [{'query': 'q'}]},
                    'call_id': 'call-1',
                }],
            },
            {
                'role': 'tool',
                'name': 'search',
                'call_id': 'call-1',
                'content': {'ok': True, 'data': {'results': []}, 'error': None, 'metadata': {}},
            },
        ]

        adapter.generate(messages)

        provider_messages = recorder.requests[0]['messages']
        self.assertEqual('call-1', provider_messages[2]['tool_calls'][0]['id'])
        self.assertEqual('call-1', provider_messages[3]['tool_call_id'])
        self.assertTrue(json.loads(provider_messages[3]['content'])['ok'])

    def test_responses_adapter_uses_only_explicit_continuation_state(self):
        function_call = SimpleNamespace(
            type='function_call',
            name='search',
            arguments='{"queries": [{"query": "q"}]}',
            call_id='call-1',
        )
        recorder = _Recorder([
            SimpleNamespace(id='response-1', output=[function_call], output_text='', usage=None),
            SimpleNamespace(id='response-2', output=[], output_text='answer', usage=None),
            SimpleNamespace(id='response-3', output=[], output_text='new answer', usage=None),
        ])
        adapter = OpenAIResponsesLLM('model', 'key', 'http://example.test/v1')
        adapter.client = SimpleNamespace(responses=recorder)
        initial_messages = [
            {'role': 'system', 'content': 'system'},
            {'role': 'user', 'content': 'question'},
        ]

        first = adapter.generate(initial_messages, [])
        continued_messages = initial_messages + [{
            'role': 'tool',
            'name': 'search',
            'call_id': 'call-1',
            'content': {'ok': True, 'data': {}, 'error': None, 'metadata': {}},
        }]
        adapter.generate(continued_messages, [], state=first.continuation)
        adapter.generate([
            {'role': 'system', 'content': 'system'},
            {'role': 'user', 'content': 'new question'},
        ], [], state=None)

        self.assertNotIn('previous_response_id', recorder.requests[0])
        self.assertEqual('response-1', recorder.requests[1]['previous_response_id'])
        self.assertEqual('function_call_output', recorder.requests[1]['input'][0]['type'])
        self.assertNotIn('previous_response_id', recorder.requests[2])
        self.assertEqual('new question', recorder.requests[2]['input'][-1]['content'])


if __name__ == '__main__':
    unittest.main()
