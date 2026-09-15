import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from local_retrieval.build_index import main as build_main, build_index
from local_retrieval.chunk_corpus import chunk_corpus
from local_retrieval.data_artifacts import artifact_pair
from local_retrieval.prepare_corpus import convert_corpus
from local_retrieval.retriever import LocalRetriever


class FakeTokenizer:
    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, tokens, **kwargs):
        return ''.join(map(chr, tokens))


class FakeEmbedder:
    name = model_name = 'fake'
    dimension = 2
    pooling = 'test'
    query_instruction = 'test'
    max_length = 512

    def encode_documents(self, texts):
        vectors = np.asarray([[text.count('apple'), text.count('banana')] for text in texts], dtype='float32')
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1)

    def encode_query(self, query):
        return self.encode_documents([query])


def build_fixture(root, manifest=None):
    chunks, index = root / 'chunks.jsonl', root / 'index.faiss'
    chunks.write_text(json.dumps({'doc_id': 'a', 'chunk_id': 0, 'text': 'apple'}) + '\n', encoding='utf-8')
    argv = ['build', '--chunks', str(chunks), '--index', str(index)]
    if manifest is not None:
        argv += ['--manifest', str(manifest)]
    with patch('sys.argv', argv), patch('local_retrieval.build_index.create_embedder', return_value=FakeEmbedder()) as factory:
        build_main()
    return index, chunks, factory


class ArtifactSafetyTest(unittest.TestCase):
    def test_index_manifest_write_failure_leaves_no_final_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('local_retrieval.build_index.save_manifest', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    build_fixture(root)
            self.assertFalse((root / 'index.faiss').exists())
            self.assertFalse((root / 'index.faiss.json').exists())
            self.assertEqual([], list(root.glob('*.tmp')))

    def test_modified_chunks_during_encoding_are_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks, index = root / 'chunks.jsonl', root / 'index.faiss'
            chunks.write_text(json.dumps({'doc_id': 'a', 'chunk_id': 0, 'text': 'apple'}) + '\n')

            class MutatingEmbedder(FakeEmbedder):
                def encode_documents(self, texts):
                    chunks.write_text(json.dumps({'doc_id': 'b', 'chunk_id': 0, 'text': 'banana'}) + '\n')
                    return super().encode_documents(texts)

            with patch('sys.argv', ['build', '--chunks', str(chunks), '--index', str(index)]), patch(
                'local_retrieval.build_index.create_embedder', return_value=MutatingEmbedder()
            ):
                with self.assertRaisesRegex(ValueError, 'changed during encoding'):
                    build_main()
            self.assertFalse(index.exists())

    def test_nonfinite_embeddings_are_rejected_after_schema_validation(self):
        class BadEmbedder(FakeEmbedder):
            def encode_documents(self, texts):
                return np.asarray([[float('nan'), 0.]], dtype='float32')

        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            build_index(iter([{'doc_id': 'a', 'chunk_id': 0, 'text': 'apple'}]), BadEmbedder(), 1)

    def test_input_output_alias_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'corpus.jsonl'
            original = json.dumps({'id': 'a', 'text': 'apple'}) + '\n'
            source.write_text(original)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                chunk_corpus(source, source, FakeTokenizer())
            self.assertEqual(original, source.read_text())

    def test_corrupt_parquet_does_not_publish_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'bad.parquet').write_bytes(b'not parquet')
            output = root / 'corpus.jsonl'
            with self.assertRaises(pa.ArrowInvalid):
                convert_corpus(root, output)
            self.assertFalse(output.exists())
            self.assertFalse(Path(f'{output}.json').exists())

    def test_predictable_old_temp_file_is_not_touched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload, metadata = root / 'data', root / 'metadata'
            old_temp = root / 'data.tmp'
            old_temp.write_text('keep')
            with artifact_pair(payload, metadata) as (data_temp, meta_temp):
                data_temp.write_text('new')
                meta_temp.write_text('metadata')
            self.assertEqual('keep', old_temp.read_text())
            self.assertEqual('new', payload.read_text())

    def test_corpus_metadata_failure_does_not_publish_any_final_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pq.write_table(pa.Table.from_pylist([{'id': 'a', 'text': 'apple'}]), root / 'part.parquet')
            output = root / 'corpus.jsonl'
            with patch.object(Path, 'write_text', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    convert_corpus(root, output)
            self.assertFalse(output.exists())
            self.assertFalse(Path(f'{output}.json').exists())
            self.assertEqual([], list(root.glob('*.tmp')))

    def test_chunks_metadata_failure_does_not_publish_any_final_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'source.jsonl', root / 'chunks.jsonl'
            source.write_text(json.dumps({'id': 'a', 'text': 'apple'}) + '\n')
            with patch.object(Path, 'write_text', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    chunk_corpus(source, output, FakeTokenizer())
            self.assertFalse(output.exists())
            self.assertFalse(Path(f'{output}.json').exists())

    def test_second_publication_failure_rolls_back_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload, metadata = root / 'data', root / 'metadata'
            original_link = os.link

            def fail_payload(source, destination):
                if destination == payload:
                    raise OSError('publication failed')
                original_link(source, destination)

            with patch('local_retrieval.data_artifacts.os.link', side_effect=fail_payload):
                with self.assertRaises(OSError):
                    with artifact_pair(payload, metadata) as (data_temp, meta_temp):
                        data_temp.write_text('data')
                        meta_temp.write_text('metadata')
            self.assertFalse(payload.exists())
            self.assertFalse(metadata.exists())

    def test_competing_writer_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload, metadata = root / 'data', root / 'metadata'
            with self.assertRaises(FileExistsError):
                with artifact_pair(payload, metadata) as (data_temp, meta_temp):
                    data_temp.write_text('ours')
                    meta_temp.write_text('ours')
                    payload.write_text('other writer')
            self.assertEqual('other writer', payload.read_text())
            self.assertFalse(metadata.exists())

    def test_existing_metadata_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / 'metadata'
            metadata.write_text('keep')
            with self.assertRaises(FileExistsError):
                with artifact_pair(root / 'data', metadata):
                    self.fail('must not enter writer')
            self.assertEqual('keep', metadata.read_text())

    def test_manifest_index_alias_is_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('local_retrieval.build_index.create_embedder') as factory, patch('sys.argv', [
                'build', '--chunks', str(root / 'chunks.jsonl'), '--index', str(root / 'index.faiss'),
                '--manifest', str(root / 'index.faiss'),
            ]):
                with self.assertRaisesRegex(ValueError, 'overlap'):
                    build_main()
                factory.assert_not_called()
            self.assertFalse((root / 'index.faiss').exists())

    def test_chunks_require_document_and_chunk_ids(self):
        for bad in ({'text': 'apple'}, {'doc_id': 'a', 'text': 'apple'},
                    {'doc_id': '', 'chunk_id': 0, 'text': 'apple'},
                    {'doc_id': 'a', 'chunk_id': -1, 'text': 'apple'}):
            with self.subTest(chunk=bad), self.assertRaises(ValueError):
                build_index(iter([bad]), FakeEmbedder(), 1)

    def test_same_count_replaced_chunks_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, chunks, _ = build_fixture(root)
            chunks.write_text(json.dumps({'doc_id': 'b', 'chunk_id': 0, 'text': 'banana'}) + '\n')
            with self.assertRaisesRegex(ValueError, 'chunks content hash'):
                LocalRetriever(index, chunks, embedder_name='fake', embedder=FakeEmbedder(), require_integrity=True)

    def test_same_shape_replaced_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, chunks, _ = build_fixture(root)
            replacement = faiss.IndexFlatIP(2)
            replacement.add(np.asarray([[0., 1.]], dtype='float32'))
            faiss.write_index(replacement, str(index))
            with self.assertRaisesRegex(ValueError, 'Index content hash'):
                LocalRetriever(index, chunks, embedder_name='fake', embedder=FakeEmbedder(), require_integrity=True)

    def test_missing_manifest_is_rejected_in_strict_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, chunks, _ = build_fixture(root)
            Path(f'{index}.json').unlink()
            with self.assertRaisesRegex(ValueError, 'manifest v2 required'):
                LocalRetriever(index, chunks, embedder_name='fake', embedder=FakeEmbedder(), require_integrity=True)
            with self.assertWarnsRegex(RuntimeWarning, 'Legacy index'):
                LocalRetriever(index, chunks, embedder_name='fake', embedder=FakeEmbedder())

    def test_parquet_to_search_with_verified_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pq.write_table(pa.Table.from_pylist([{'id': 'a', 'text': 'apple'}, {'id': 'b', 'text': 'banana'}]), root / 'part.parquet')
            corpus, chunks, index = root / 'corpus.jsonl', root / 'chunks.jsonl', root / 'index.faiss'
            convert_corpus(root, corpus, expected_docs=2)
            chunk_corpus(corpus, chunks, FakeTokenizer())
            with patch('sys.argv', ['build', '--chunks', str(chunks), '--index', str(index)]), patch('local_retrieval.build_index.create_embedder', return_value=FakeEmbedder()):
                build_main()
            retriever = LocalRetriever(index, chunks, embedder_name='fake', embedder=FakeEmbedder(), require_integrity=True)
            self.assertEqual('a', retriever.search('apple', 1)[0]['doc_id'])
            self.assertEqual('b', retriever.search('banana', 1)[0]['doc_id'])


if __name__ == '__main__':
    unittest.main()
