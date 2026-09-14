import json
import tempfile
import unittest
from pathlib import Path

from my_ReAct.experiment import ensure_experiment
from my_ReAct.run import save_run, completed_query_ids


class ExperimentTest(unittest.TestCase):
    def test_resume_requires_same_config_and_preserves_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            identity = ensure_experiment(path, {'model': 'a', 'context': None})
            result = {'query_id': '1', 'status': 'completed', 'experiment_id': identity}
            save_run(result, path)
            self.assertEqual(identity, ensure_experiment(path, {'model': 'a', 'context': None}))
            self.assertEqual(completed_query_ids(path), {'1'})
            for config in ({'model': 'b', 'context': None}, {'model': 'a', 'context': 4}):
                with self.assertRaisesRegex(ValueError, 'configuration changed'):
                    ensure_experiment(path, config)
            self.assertEqual(json.loads((path / 'run_1.json').read_text()), result)
            self.assertFalse((path / 'run_1.json.tmp').exists())

    def test_legacy_directory_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_run({'query_id': '1', 'status': 'completed'}, path)
            with self.assertRaisesRegex(ValueError, 'Legacy'):
                ensure_experiment(path, {'model': 'a'})

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save_run({'query_id': '../../escape'}, Path(directory))
