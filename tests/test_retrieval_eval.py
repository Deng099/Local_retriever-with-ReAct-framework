import tempfile
import unittest
from pathlib import Path

from local_retrieval.retrieval_eval import evaluate, load_qrels, load_queries, save_report


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search_batch(self, queries, top_ks):
        self.calls.append((queries, top_ks))
        return [self.results[query][:top_k] for query, top_k in zip(queries, top_ks)]


class RetrievalEvalTest(unittest.TestCase):
    def test_loads_queries_and_positive_qrels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            query_path = temp_path / 'queries.tsv'
            query_path.write_text('q1\tfirst query\nq2\tsecond query\n', encoding='utf-8')
            qrel_path = temp_path / 'qrels.txt'
            qrel_path.write_text(
                'q1 Q0 d1 1\nq1 Q0 ignored 0\nq2 Q0 d2 2\n',
                encoding='utf-8',
            )
            self.assertEqual(
                load_queries(query_path),
                [('q1', 'first query'), ('q2', 'second query')],
            )
            self.assertEqual(load_qrels(qrel_path), {'q1': {'d1'}, 'q2': {'d2'}})

    def test_evaluates_evidence_recall_and_duplicate_chunks(self):
        retriever = FakeRetriever({
            'find it': [
                {'doc_id': 'noise', 'chunk_id': 0, 'score': 0.9, 'text': 'x'},
                {'doc_id': 'd1', 'chunk_id': 0, 'score': 0.8, 'text': 'a'},
                {'doc_id': 'd1', 'chunk_id': 1, 'score': 0.7, 'text': 'b'},
                {'doc_id': 'd2', 'chunk_id': 0, 'score': 0.6, 'text': 'c'},
            ],
        })
        report = evaluate(
            retriever,
            [('q1', 'find it')],
            {'q1': {'d1', 'd2'}},
            cutoffs=(2, 4),
            batch_size=1,
        )
        result = report['results'][0]
        self.assertEqual(result['metrics']['recall@2'], 0.5)
        self.assertEqual(result['metrics']['recall@4'], 1.0)
        self.assertEqual(result['duplicate_chunk_count'], 1)
        self.assertEqual(result['unique_documents_at_cutoff'], {'2': 2, '4': 3})
        self.assertEqual(result['relevant_chunk_ranks'], {'d1': 2, 'd2': 4})
        self.assertEqual(result['ranked_docids'], ['noise', 'd1', 'd2'])
        self.assertGreater(result['metrics']['ndcg@10'], 0)
        self.assertEqual(retriever.calls, [(['find it'], [4])])

    def test_records_batch_failure_and_saves_atomically(self):
        class FailingRetriever:
            def search_batch(self, queries, top_ks):
                raise RuntimeError('offline')

        report = evaluate(
            FailingRetriever(),
            [('q1', 'query')],
            {'q1': {'d1'}},
            cutoffs=(5,),
        )
        self.assertEqual(report['summary']['failed_count'], 1)
        self.assertIn('offline', report['results'][0]['error'])
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'report.json'
            save_report(report, output_path)
            self.assertTrue(output_path.exists())
            self.assertFalse(output_path.with_suffix('.json.tmp').exists())


if __name__ == '__main__':
    unittest.main()
