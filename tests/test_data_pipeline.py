import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from local_retrieval.prepare_corpus import convert_corpus
from local_retrieval.chunk_corpus import chunk_corpus
from local_retrieval.build_index import build_index, load_chunks


class FakeTokenizer:
    def encode(self, text, **kwargs):
        return list(range(len(text)))

    def decode(self, tokens, **kwargs):
        return ' '.join(str(token) for token in tokens)


class DataPipelineTest(unittest.TestCase):
    def test_parquet_conversion_preserves_sources_and_reports_empty_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pq.write_table(pa.Table.from_pylist([
                {'docid': 'a', 'text': 'first', 'url': 'https://example.test/a'},
                {'docid': 'b', 'text': ' ', 'url': None},
            ]), root / 'train-0.parquet')
            pq.write_table(pa.Table.from_pylist([
                {'id': 'c', 'text': 'second', 'title': 'Article'},
            ]), root / 'train-1.parquet')
            output = root / 'corpus.jsonl'
            stats = convert_corpus(root, output, batch_size=1, expected_docs=2)
            rows = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(['a', 'c'], [row['id'] for row in rows])
            self.assertEqual('https://example.test/a', rows[0]['url'])
            self.assertEqual('Article', rows[1]['title'])
            self.assertEqual(1, stats['empty_text'])
            with self.assertRaises(FileExistsError):
                convert_corpus(root, output)

    def test_duplicate_ids_do_not_publish_final_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pq.write_table(pa.Table.from_pylist([
                {'id': 'a', 'text': 'one'}, {'id': 'a', 'text': 'two'},
            ]), root / 'train.parquet')
            output = root / 'corpus.jsonl'
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                convert_corpus(root, output)
            self.assertFalse(output.exists())

    def test_chunking_records_parameters_and_document_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / 'corpus.jsonl'
            corpus.write_text(json.dumps({'id': 'a', 'text': 'abcdef'}) + '\n', encoding='utf-8')
            output = root / 'chunks.jsonl'
            stats = chunk_corpus(corpus, output, FakeTokenizer(), chunk_size=4, overlap=1)
            self.assertEqual(2, stats['chunks'])
            self.assertEqual(['0 1 2 3', '3 4 5'], [row['text'] for row in load_chunks(output)])
            metadata = json.loads(Path(f'{output}.json').read_text())
            self.assertEqual(1, metadata['chunk_overlap'])
            with self.assertRaises(FileExistsError):
                chunk_corpus(corpus, output, FakeTokenizer())

    def test_index_consumes_one_pass_iterator_in_bounded_batches(self):
        class FakeEmbedder:
            dimension = 2

            def __init__(self):
                self.batch_sizes = []

            def encode_documents(self, texts):
                self.batch_sizes.append(len(texts))
                return np.asarray([[1.0, 0.0]] * len(texts), dtype='float32')

        embedder = FakeEmbedder()
        chunks = ({'text': str(index)} for index in range(5))
        index = build_index(chunks, embedder, batch_size=2)
        self.assertEqual([2, 2, 1], embedder.batch_sizes)
        self.assertEqual(5, index.ntotal)

    def test_empty_index_is_rejected(self):
        class FakeEmbedder:
            dimension = 2

        with self.assertRaisesRegex(ValueError, 'No chunks'):
            build_index(iter([]), FakeEmbedder(), batch_size=2)


if __name__ == '__main__':
    unittest.main()
