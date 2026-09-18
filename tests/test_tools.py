import os
import json
import unittest

from my_ReAct.base_tool import BaseTool, RegisterTool
from my_ReAct.python_tool import PythonTool
from my_ReAct.run import collect_run_data, collect_timing_data
from my_ReAct.prompts import DEFAULT_TOOL_USE_POLICY
from my_ReAct.python_session import (
    PythonSessionCrashed,
    PythonSessionManager,
    PythonSessionNotFound,
    PythonSessionTimeout,
)
from my_ReAct.react import LLMResult, ReActAgent, ToolCall
from my_ReAct.search_tool import SearchTool
from my_ReAct.toolset import ResearchTools
from my_ReAct.visit_tool import VisitTool


class EchoTool(BaseTool):
    def __init__(self):
        super().__init__('echo', 'Return exact input content.', {
            'type': 'object',
            'properties': {'text': {'type': 'string'}},
            'required': ['text'],
        })

    def _execute(self, text):
        return text


class ToolTest(unittest.TestCase):
    def test_agent_reports_empty_answer(self):
        class EmptyLLM:
            def generate(self, messages, tools=None, state=None):
                return LLMResult()

        class FakeRetriever:
            def search_batch(self, queries, top_ks):
                return [[] for _ in queries]

            def visit(self, reference, goal=None):
                return {'doc_id': reference, 'content': 'text'}

        retriever = FakeRetriever()
        result = ReActAgent(
            llm=EmptyLLM(),
            max_step=1,
            query='question',
            tools=ResearchTools(
                search=SearchTool(retriever),
                visit=VisitTool(retriever),
                python=PythonTool(),
            ),
        ).run()

        self.assertEqual('incomplete', result.status)
        self.assertEqual('empty_answer', result.termination_reason)

    def test_agent_reports_tool_budget_exhaustion(self):
        class BudgetLLM:
            def __init__(self):
                self.calls = 0

            def generate(self, messages, tools=None, state=None):
                self.calls += 1
                if self.calls == 1:
                    return LLMResult(tool_calls=[ToolCall(
                        'search',
                        {'queries': [{'query': 'q', 'top_k': 1}]},
                    )])
                return LLMResult(text='answer from evidence')

        class FakeRetriever:
            def search_batch(self, queries, top_ks):
                return [[] for _ in queries]

            def visit(self, reference, goal=None):
                return {'doc_id': reference, 'content': 'text'}

        retriever = FakeRetriever()
        result = ReActAgent(
            llm=BudgetLLM(),
            max_step=1,
            query='question',
            tools=ResearchTools(
                search=SearchTool(retriever),
                visit=VisitTool(retriever),
                python=PythonTool(),
            ),
        ).run()

        self.assertEqual('completed', result.status)
        self.assertEqual('tool_budget_exhausted', result.termination_reason)
        self.assertEqual('answer from evidence', result.output)

    def test_runner_collects_search_data_from_tool_result_envelope(self):
        messages = [
            {
                'role': 'assistant',
                'tool_calls': [{'name': 'search'}, {'name': 'python'}],
            },
            {
                'role': 'tool',
                'name': 'search',
                'content': {
                    'ok': True,
                    'data': {'results': [
                        {'query': 'q', 'chunks': [
                            {'doc_id': 'doc-1'},
                            {'doc_id': 'doc-1'},
                            {'doc_id': 'doc-2'},
                        ]},
                    ]},
                    'error': None,
                    'metadata': {},
                },
            },
        ]

        counts, docids = collect_run_data(messages)

        self.assertEqual({'search': 1, 'python': 1}, counts)
        self.assertEqual(['doc-1', 'doc-2'], docids)

    def test_runner_collects_llm_and_tool_timing(self):
        timing = collect_timing_data([
            {'role': 'assistant', 'latency_seconds': 2.5},
            {'role': 'tool', 'name': 'search', 'latency_seconds': 0.4},
            {'role': 'tool', 'name': 'search', 'latency_seconds': 0.6},
            {'role': 'tool', 'name': 'visit', 'latency_seconds': 0.2},
            {'role': 'assistant', 'latency_seconds': -1},
        ])

        self.assertEqual(2.5, timing['llm_seconds'])
        self.assertAlmostEqual(1.2, timing['tool_seconds'])
        self.assertEqual({'search': 1.0, 'visit': 0.2}, timing['tool_seconds_by_name'])

    def test_default_tool_policy_requires_evidence_sufficiency_and_no_repeat_search(self):
        self.assertIn('specific unresolved fact', DEFAULT_TOOL_USE_POLICY)
        self.assertIn('semantically equivalent search', DEFAULT_TOOL_USE_POLICY)
        self.assertIn('stop using tools and answer', DEFAULT_TOOL_USE_POLICY)

    def test_tool_registry_returns_one_serializable_result_contract(self):
        registry = RegisterTool([EchoTool()])

        success = registry.execute('echo', text='hello')
        invalid_arguments = registry.execute('echo', wrong='value')
        unknown = registry.execute('missing')

        self.assertEqual({'ok', 'data', 'error', 'metadata'}, set(success))
        self.assertTrue(success['ok'])
        self.assertEqual('hello', success['data'])
        self.assertFalse(invalid_arguments['ok'])
        self.assertEqual('TypeError', invalid_arguments['error']['type'])
        self.assertFalse(unknown['ok'])
        self.assertEqual('UnknownTool', unknown['error']['type'])
        json.dumps([success, invalid_arguments, unknown])

    def test_agent_closes_python_session_when_llm_fails(self):
        class FailingLLM:
            def generate(self, messages, tools=None, state=None):
                raise RuntimeError('simulated LLM failure')

        class FakeRetriever:
            def search_batch(self, queries, top_ks):
                return [[] for _ in queries]

            def visit(self, reference, goal=None):
                return {'doc_id': reference, 'content': 'text'}

        python_tool = PythonTool()
        agent = ReActAgent(
            llm=FailingLLM(),
            max_step=1,
            query='question',
            tools=ResearchTools(
                search=SearchTool(FakeRetriever()),
                visit=VisitTool(FakeRetriever()),
                python=python_tool,
            ),
        )

        result = agent.run()

        self.assertEqual('error', result.status)
        self.assertEqual('runtime_error', result.termination_reason)
        self.assertEqual('RuntimeError', result.error.type)
        self.assertFalse(python_tool.is_active)

    def test_python_session_manager_cleans_up_idle_sessions(self):
        with PythonSessionManager(execution_timeout=5) as manager:
            session_id = manager.create_session()

            cleaned = manager.cleanup_idle(max_idle_seconds=0)

            self.assertEqual([session_id], cleaned)
            self.assertFalse(manager.is_active(session_id))

    def test_python_session_manager_preserves_and_isolates_state(self):
        with PythonSessionManager(execution_timeout=5) as manager:
            first = manager.create_session('first')
            second = manager.create_session('second')

            self.assertEqual('ok', manager.execute(first, 'a = 10')['status'])
            self.assertEqual('15', manager.execute(first, 'a + 5')['result'])
            isolated = manager.execute(second, 'a')

            self.assertEqual('error', isolated['status'])
            self.assertEqual('NameError', isolated['error']['type'])
            self.assertNotEqual(os.getpid(), manager.worker_pid(first))

    def test_python_session_manager_returns_packages_and_output(self):
        with PythonSessionManager(execution_timeout=5) as manager:
            session_id = manager.create_session()
            result = manager.execute(
                session_id,
                "print(math.sqrt(16))\n(float(np.mean([1, 2, 3])), int(pd.Series([1]).sum()), scipy.__name__)",
            )

            self.assertEqual('ok', result['status'])
            self.assertEqual('4.0\n', result['stdout'])
            self.assertIn("(2.0, 1, 'scipy')", result['result'])

    def test_python_output_is_bounded_inside_worker(self):
        with PythonSessionManager(
            execution_timeout=5,
            stdout_limit=10,
            stderr_limit=8,
            result_limit=12,
        ) as manager:
            session_id = manager.create_session()
            result = manager.execute(
                session_id,
                "import sys\nprint('x' * 100)\nprint('e' * 100, file=sys.stderr)\n'y' * 100",
            )

            self.assertEqual(10, len(result['stdout']))
            self.assertEqual(8, len(result['stderr']))
            self.assertLessEqual(len(result['result']), 12)
            self.assertEqual(
                {'stdout': True, 'stderr': True, 'result': True},
                result['truncated'],
            )

    def test_python_session_timeout_closes_worker(self):
        with PythonSessionManager(execution_timeout=0.2) as manager:
            session_id = manager.create_session()

            with self.assertRaises(PythonSessionTimeout):
                manager.execute(session_id, 'while True: pass')
            with self.assertRaises(PythonSessionNotFound):
                manager.execute(session_id, '1 + 1')

    def test_python_session_crash_is_detected_and_removed(self):
        with PythonSessionManager(execution_timeout=5) as manager:
            session_id = manager.create_session()

            with self.assertRaises(PythonSessionCrashed):
                manager.execute(session_id, 'import os; os._exit(7)')
            with self.assertRaises(PythonSessionNotFound):
                manager.execute(session_id, '1 + 1')

    def test_agent_owns_fixed_tools_and_preserves_python_state(self):
        class FakeRetriever:
            def search_batch(self, queries, top_ks):
                return [[{
                    'doc_id': 'doc-1',
                    'chunk_id': 'doc-1-0',
                    'score': 1.0,
                    'text': 'snippet',
                }] for _ in queries]

            def visit(self, reference, goal=None):
                return {'doc_id': reference, 'content': 'full text'}

        class FakeLLM:
            def __init__(self):
                self.step = 0

            def generate(self, messages, tools=None, state=None):
                calls = [
                    ToolCall('search', {'queries': [{'query': 'question', 'top_k': 1}]}),
                    ToolCall('visit', {'reference': 'doc-1'}),
                    ToolCall('python', {'code': 'a = 10'}),
                    ToolCall('python', {'code': 'a + 5'}),
                ]
                if self.step < len(calls):
                    call = calls[self.step]
                    self.step += 1
                    return LLMResult(tool_calls=[call])
                return LLMResult(text='done')

        retriever = FakeRetriever()
        python_tool = PythonTool()
        agent = ReActAgent(
            llm=FakeLLM(),
            max_step=5,
            query='question',
            tools=ResearchTools(
                search=SearchTool(retriever),
                visit=VisitTool(retriever),
                python=python_tool,
            ),
        )

        run_result = agent.run()
        self.assertEqual('done', run_result.output)
        self.assertEqual('completed', run_result.status)
        self.assertEqual('final_answer', run_result.termination_reason)
        self.assertEqual(['search', 'visit', 'python'], [tool['function']['name'] for tool in agent.tool_schemas])
        self.assertEqual('system', agent.messages[0]['role'])
        python_results = [
            message['content']
            for message in agent.messages
            if message['role'] == 'tool' and message['name'] == 'python'
        ]
        self.assertTrue(python_results[-1]['ok'])
        self.assertEqual('15', python_results[-1]['data']['result'])
        self.assertFalse(python_tool.is_active)
        timed_messages = [
            message for message in agent.messages
            if message['role'] in {'assistant', 'tool'}
        ]
        self.assertTrue(timed_messages)
        self.assertTrue(all(message.get('latency_seconds', -1) >= 0 for message in timed_messages))
        python_schema = agent.tool_schemas[-1]['function']['parameters']
        self.assertNotIn('session_id', python_schema['properties'])

    def test_python_variables_persist_within_one_tool(self):
        with PythonTool() as tool:
            tool.execute('a = 10')

            result = tool.execute('a + 5')
            self.assertTrue(result.ok)
            self.assertEqual('15', result.data['result'])

    def test_python_namespaces_are_isolated(self):
        with PythonTool() as first, PythonTool() as second:
            first.execute('a = 10')

            result = second.execute('a')
            self.assertFalse(result.ok)
            self.assertEqual('NameError', result.error.type)

    def test_visit_delegates_to_fixed_corpus_client(self):
        class FakeVisitor:
            def visit(self, reference, goal=None):
                return {
                    'source': {'type': 'corpus', 'doc_id': reference, 'url': None},
                    'document': {'content': 'full text'},
                    'extraction': {'goal': goal, 'content': None} if goal else None,
                    'cache': {'document_hit': True, 'extraction_hit': None},
                }

        result = VisitTool(FakeVisitor()).execute('doc-1', goal='find the answer')

        self.assertTrue(result.ok)
        self.assertEqual('full text', result.data['document']['content'])
        self.assertEqual('find the answer', result.data['extraction']['goal'])
        self.assertIn('goal', VisitTool(FakeVisitor()).parameters['properties'])


if __name__ == '__main__':
    unittest.main()
