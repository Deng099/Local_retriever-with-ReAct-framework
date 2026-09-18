import json
import tempfile
import unittest
from pathlib import Path

from my_ReAct.inspect_run import find_run_paths, render_run


class InspectRunTest(unittest.TestCase):
    def test_render_run_includes_labels_metrics_and_compact_trajectory(self):
        run = {
            'query_id': 'q1',
            'status': 'completed',
            'termination_reason': 'final_answer',
            'tool_call_counts': {'search': 1},
            'retrieved_docids': ['doc-1', 'other'],
            'result': [{'type': 'output_text', 'output': 'Exact Answer: answer'}],
            'run_state': {'step': 2, 'tool_rounds': 1, 'elapsed': 3.0},
            'usage': [{'prompt_tokens': 10, 'completion_tokens': 2}],
            'timing': {'llm_seconds': 2.0, 'tool_seconds': 0.5},
            'trajectory': [
                {'role': 'assistant', 'tool_calls': [{'name': 'search'}], 'latency_seconds': 1.0},
                {'role': 'tool', 'name': 'search', 'content': {'large': 'x' * 50}, 'latency_seconds': 0.5},
                {'role': 'assistant', 'content': 'Exact Answer: answer', 'latency_seconds': 1.0},
            ],
        }

        report = render_run(
            run,
            {'q1': {'answer': 'answer'}},
            {'q1': {'evidence': ['doc-1', 'doc-2'], 'gold': ['doc-1']}},
            show_trajectory=True,
            max_chars=20,
        )

        self.assertIn('evidence recall: 1/2 (50.0%)', report)
        self.assertIn('gold recall: 1/1 (100.0%)', report)
        self.assertIn('tokens (prompt/completion): 10/2', report)
        self.assertIn('GROUND TRUTH:\nanswer', report)
        self.assertIn('TRAJECTORY:', report)
        self.assertIn('...', report)

    def test_find_run_paths_filters_query_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'run_a.json').write_text('{}', encoding='utf-8')
            (root / 'run_b.json').write_text('{}', encoding='utf-8')

            self.assertEqual([root / 'run_b.json'], find_run_paths(root, 'b'))


if __name__ == '__main__':
    unittest.main()
