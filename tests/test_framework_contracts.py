import os
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from my_ReAct.base_tool import BaseTool
from my_ReAct.contracts import AgentRunResult
from my_ReAct.run import run_query
from my_ReAct.settings import AgentSettings, LLMSettings
from my_ReAct.toolset import ResearchTools
from my_ReAct.run_state import RunState


class EchoTool(BaseTool):
    def __init__(self):
        super().__init__('echo', 'Return exact input content.', {
            'type': 'object',
            'properties': {'text': {'type': 'string'}},
            'required': ['text'],
        })

    def _execute(self, text):
        return text


class FrameworkContractTest(unittest.TestCase):
    def test_llm_settings_are_validated_and_do_not_repr_api_key(self):
        environment = {
            'OPENAI_MODEL': 'test-model',
            'OPENAI_API_KEY': 'secret-value',
            'OPENAI_BASE_URL': 'https://example.test/v1',
            'OPENAI_API_TYPE': 'responses',
            'OPENAI_TRUST_ENV': 'false',
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = LLMSettings.from_env()

        self.assertEqual('test-model', settings.model)
        self.assertEqual('responses', settings.api_type)
        self.assertFalse(settings.trust_env)
        self.assertNotIn('secret-value', repr(settings))

    def test_agent_settings_reject_invalid_limits(self):
        with self.assertRaises(ValueError):
            AgentSettings(max_steps=0)
        with self.assertRaises(ValueError):
            AgentSettings(python_execution_timeout=0)
        with self.assertRaises(ValueError):
            AgentSettings(python_output_limit=0)

    def test_research_tools_enforces_fixed_roles(self):
        echo = EchoTool()

        with self.assertRaisesRegex(ValueError, 'requires search, visit, python'):
            ResearchTools(search=echo, visit=echo, python=echo)

    def test_pyproject_declares_packages_extras_and_cli(self):
        project_root = Path(__file__).resolve().parents[1]
        with (project_root / 'pyproject.toml').open('rb') as file:
            pyproject = tomllib.load(file)

        self.assertEqual('local-react-rag', pyproject['project']['name'])
        self.assertIn('server', pyproject['project']['optional-dependencies'])
        self.assertNotIn('vllm', ' '.join(pyproject['project']['optional-dependencies']['server']).lower())
        self.assertEqual('my_ReAct.run:main', pyproject['project']['scripts']['react-rag'])
        self.assertEqual(
            'my_ReAct.inspect_run:main',
            pyproject['project']['scripts']['react-rag-inspect'],
        )

    @patch('my_ReAct.run.create_research_agent')
    @patch('my_ReAct.run.create_llm')
    def test_runner_serializes_agent_status_and_termination_reason(
        self,
        create_llm,
        create_research_agent,
    ):
        agent = unittest.mock.Mock()
        agent.messages = [{'role': 'user', 'content': 'formatted question'}]
        agent.state = RunState(step=2, tool_rounds=1)
        agent.state.finish('tool_budget_exhausted')
        agent.run.return_value = AgentRunResult.incomplete('tool_budget_exhausted')
        create_llm.return_value = object()
        create_research_agent.return_value = agent
        settings = LLMSettings(model='test-model', api_key='secret')

        record = run_query('q1', 'question', object(), 1, llm_settings=settings,
                           context_turns=4, retrieval_info={'embedder': 'custom'})

        self.assertEqual('incomplete', record['status'])
        self.assertEqual('tool_budget_exhausted', record['termination_reason'])
        self.assertEqual('', record['result'][0]['output'])
        self.assertEqual(2, record['run_state']['step'])
        self.assertEqual('custom', record['embedder'])
        self.assertEqual(4, create_research_agent.call_args.kwargs['context_manager'].keep_recent_turns)


if __name__ == '__main__':
    unittest.main()
