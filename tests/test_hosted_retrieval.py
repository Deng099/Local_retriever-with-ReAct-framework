import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import faiss
import numpy as np
from fastapi.testclient import TestClient

from local_retrieval.retriever import LocalRetriever
from local_retrieval.document_store import CorpusDocumentStore
from local_retrieval.retrieval_client import RetrievalClient
from local_retrieval.retrieval_service import create_app
from local_retrieval.web_document_store import (
    WebDocumentStore,
    WebVisitError,
    URLAccessPolicy,
    _PolicyRedirectHandler,
    extract_main_content,
    fetch_html,
    normalize_url,
)
from local_retrieval.visit_extraction import VisitExtractionCache
from my_ReAct.search_tool import SearchTool


class FakeBatchEmbedder:
    name = 'fake'
    model_name = 'fake-embedder'
    dimension = 2

    def encode_queries(self, queries):
        vectors = {
            'first': [1.0, 0.0],
            'second': [0.0, 1.0],
        }
        return np.asarray([vectors[query] for query in queries], dtype='float32')


class HostedRetrievalTest(unittest.TestCase):
    def test_retrieval_client_sends_private_token(self):
        client = RetrievalClient(api_key='secret')
        self.assertEqual('Bearer secret', client._headers()['Authorization'])

    def test_http_service_can_require_private_token(self):
        class FakeRetriever:
            index = SimpleNamespace(ntotal=2, d=2)
            embedder_name = 'fake'

        client = TestClient(create_app(FakeRetriever(), api_key='secret'))

        self.assertEqual(401, client.get('/health').status_code)
        self.assertEqual(
            200,
            client.get('/health', headers={'Authorization': 'Bearer secret'}).status_code,
        )

    def test_redirect_handler_revalidates_destination(self):
        policy = URLAccessPolicy()
        handler = _PolicyRedirectHandler(policy)
        private_address = [
            (2, 1, 6, '', ('127.0.0.1', 80)),
        ]

        with patch(
            'local_retrieval.web_document_store.socket.getaddrinfo',
            return_value=private_address,
        ):
            with self.assertRaisesRegex(WebVisitError, 'non-public address'):
                handler.redirect_request(
                    None,
                    None,
                    302,
                    'Found',
                    {},
                    'http://redirect-target.test/private',
                )
    def test_http_service_accepts_batch_search_and_visit_goal(self):
        class FakeRetriever:
            index = SimpleNamespace(ntotal=2, d=2)
            embedder_name = 'fake'

            def search_batch(self, queries, top_ks):
                return [[{
                    'doc_id': f'doc-{index}',
                    'chunk_id': f'chunk-{index}',
                    'score': 1.0,
                    'text': query,
                }] * top_k for index, (query, top_k) in enumerate(zip(queries, top_ks))]

        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_path = Path(temp_dir) / 'corpus.jsonl'
            corpus_path.write_text(
                json.dumps({'id': 'doc-1', 'text': 'fixed evidence'}) + '\n',
                encoding='utf-8',
            )
            client = TestClient(create_app(FakeRetriever(), CorpusDocumentStore(corpus_path)))

            search_response = client.post('/search', json={'queries': [
                {'query': 'first', 'top_k': 2},
                {'query': 'second', 'top_k': 1},
            ]})
            visit_response = client.post('/visit', json={
                'reference': 'doc-1',
                'goal': 'find evidence',
            })

            self.assertEqual(200, search_response.status_code)
            self.assertEqual([2, 1], [len(item['chunks']) for item in search_response.json()['results']])
            self.assertEqual(200, visit_response.status_code)
            self.assertEqual('not_configured', visit_response.json()['extraction']['status'])
            self.assertEqual('fixed evidence', visit_response.json()['document']['content'])

    def test_real_static_html_fetch_clean_and_non_html_error(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/slow':
                    time.sleep(0.2)
                    body = b'<html><body>late response</body></html>'
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html')
                elif self.path == '/plain':
                    body = b'not html'
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain')
                else:
                    body = b'''<html><head><title>Local article</title></head><body>
                        <nav>Menu</nav><article><h1>Verified heading</h1>
                        <p>This locally served article contains meaningful evidence for a
                        deterministic end-to-end web extraction test.</p></article></body></html>'''
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except BrokenPipeError:
                    pass

            def log_message(self, format, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                store = WebDocumentStore(
                    Path(temp_dir) / 'visit.sqlite3',
                    allow_private_network=True,
                )
                base_url = f'http://127.0.0.1:{server.server_port}'

                with self.assertRaisesRegex(WebVisitError, 'non-public address'):
                    fetch_html(f'{base_url}/article')
                document = store.visit(f'{base_url}/article')
                with self.assertRaises(WebVisitError):
                    store.visit(f'{base_url}/plain')
                with self.assertRaisesRegex(WebVisitError, 'timed out'):
                    fetch_html(
                        f'{base_url}/slow',
                        timeout=0.01,
                        allow_private_network=True,
                    )

                empty_store = WebDocumentStore(
                    Path(temp_dir) / 'empty.sqlite3',
                    fetcher=lambda url: {
                        'final_url': url,
                        'status_code': 200,
                        'content_type': 'text/html',
                        'html': '<html></html>',
                    },
                    extractor=lambda html: {'title': None, 'content': '   '},
                )
                with self.assertRaisesRegex(WebVisitError, 'No readable main content'):
                    empty_store.visit(f'{base_url}/empty')

                self.assertEqual('Local article', document['title'])
                self.assertIn('meaningful evidence', document['content'])
                self.assertFalse(document['cache_hit'])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_goal_extraction_uses_content_goal_model_and_prompt_cache_key(self):
        class FakeExtractor:
            model = 'fake-model'
            prompt_version = 'prompt-v1'

            def __init__(self):
                self.calls = []

            def extract(self, document, goal):
                self.calls.append((document['content'], goal))
                return f'Extracted for {goal}'

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            corpus_path = temp_path / 'corpus.jsonl'
            cache_path = temp_path / 'visit.sqlite3'
            corpus_path.write_text(
                json.dumps({'id': 'doc-1', 'url': 'https://example.test/1', 'text': 'full fixed text'}) + '\n',
                encoding='utf-8',
            )
            extractor = FakeExtractor()
            first_store = CorpusDocumentStore(
                corpus_path,
                goal_extractor=extractor,
                extraction_cache=VisitExtractionCache(cache_path),
            )
            first = first_store.visit('doc-1', goal='find date')
            second_store = CorpusDocumentStore(
                corpus_path,
                goal_extractor=extractor,
                extraction_cache=VisitExtractionCache(cache_path),
            )
            second = second_store.visit('doc-1', goal='find   date')
            third = second_store.visit('doc-1', goal='find person')

            self.assertEqual('full fixed text', first['document']['content'])
            self.assertEqual('Extracted for find date', first['extraction']['content'])
            self.assertFalse(first['extraction']['cache_hit'])
            self.assertTrue(second['extraction']['cache_hit'])
            self.assertFalse(third['extraction']['cache_hit'])
            self.assertEqual(2, len(extractor.calls))

    def test_goal_visit_without_extractor_returns_cleaned_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_path = Path(temp_dir) / 'corpus.jsonl'
            corpus_path.write_text(
                json.dumps({'id': 'doc-1', 'text': 'full fixed text'}) + '\n',
                encoding='utf-8',
            )
            result = CorpusDocumentStore(corpus_path).visit('doc-1', goal='find date')

            self.assertEqual('full fixed text', result['document']['content'])
            self.assertEqual('not_configured', result['extraction']['status'])

    def test_long_visit_returns_bounded_view_and_supports_range_reads(self):
        class FakeExtractor:
            model = 'fake-model'
            prompt_version = 'prompt-v1'
            max_input_chars = 50

            def extract(self, document, goal):
                return f'evidence for {goal}'

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            corpus_path = temp_path / 'corpus.jsonl'
            corpus_path.write_text(
                json.dumps({'id': 'doc-1', 'text': '0123456789' * 20}) + '\n',
                encoding='utf-8',
            )
            store = CorpusDocumentStore(
                corpus_path,
                inline_content_limit=12,
                goal_extractor=FakeExtractor(),
                extraction_cache=VisitExtractionCache(temp_path / 'visit.sqlite3'),
            )

            extracted = store.visit('doc-1', goal='find dates')
            self.assertEqual('', extracted['document']['content'])
            self.assertTrue(extracted['document']['content_omitted'])
            self.assertTrue(extracted['document']['content_truncated'])
            self.assertEqual('evidence for find dates', extracted['extraction']['content'])
            self.assertTrue(extracted['extraction']['input_truncated'])

            ranged = store.visit('doc-1', offset=20, limit=5)
            self.assertEqual('01234', ranged['document']['content'])
            self.assertEqual(20, ranged['document']['content_offset'])
            self.assertEqual(25, ranged['document']['next_offset'])

    def test_trafilatura_extracts_title_and_main_text(self):
        extracted = extract_main_content('''
            <html><head><title>Research result</title></head><body>
            <nav>Navigation links that are not the article.</nav>
            <article><h1>Important finding</h1>
            <p>This is the useful research content. It contains enough detail for the
            visitor to return meaningful evidence instead of navigation boilerplate.</p>
            <p>A second paragraph records the supporting facts and conclusions.</p>
            </article><footer>Copyright notice</footer></body></html>
        ''')

        self.assertEqual('Research result', extracted['title'])
        self.assertIn('useful research content', extracted['content'])

    def test_live_web_visit_is_cleaned_and_persists_across_store_instances(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            corpus_path = temp_path / 'corpus.jsonl'
            cache_path = temp_path / 'visit.sqlite3'
            corpus_path.write_text('', encoding='utf-8')
            fetch_count = {'value': 0}

            def fetcher(url):
                fetch_count['value'] += 1
                return {
                    'final_url': url,
                    'status_code': 200,
                    'content_type': 'text/html',
                    'html': '<html><title>Example</title><body><nav>Menu</nav><main>Useful text</main></body></html>',
                }

            def extractor(html):
                self.assertIn('Useful text', html)
                return {'title': 'Example', 'content': 'Useful text'}

            first_store = CorpusDocumentStore(
                corpus_path,
                web_store=WebDocumentStore(cache_path, fetcher=fetcher, extractor=extractor),
            )
            first = first_store.visit('HTTPS://EXAMPLE.TEST:443/page#fragment')
            second_store = CorpusDocumentStore(
                corpus_path,
                web_store=WebDocumentStore(cache_path, fetcher=fetcher, extractor=extractor),
            )
            second = second_store.visit('https://example.test/page')

            self.assertFalse(first['cache']['document_hit'])
            self.assertTrue(second['cache']['document_hit'])
            self.assertEqual('https://example.test/page', first['source']['normalized_url'])
            self.assertEqual('Useful text', second['document']['content'])
            self.assertEqual(
                first['document']['content_hash'],
                second['document']['content_hash'],
            )
            self.assertEqual(1, fetch_count['value'])

    def test_fixed_corpus_url_never_calls_live_web(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            corpus_path = temp_path / 'corpus.jsonl'
            corpus_url = 'https://example.test/fixed'
            corpus_path.write_text(
                json.dumps({'id': 'doc-1', 'url': corpus_url, 'text': 'fixed text'}) + '\n',
                encoding='utf-8',
            )

            def fail_fetcher(url):
                self.fail(f'fixed corpus unexpectedly fetched {url}')

            store = CorpusDocumentStore(
                corpus_path,
                web_store=WebDocumentStore(temp_path / 'visit.sqlite3', fetcher=fail_fetcher),
            )

            self.assertEqual(
                'fixed text',
                store.visit('HTTPS://EXAMPLE.TEST:443/fixed#fragment')['document']['content'],
            )

    def test_web_visit_rejects_non_http_urls(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = WebDocumentStore(Path(temp_dir) / 'visit.sqlite3')

            with self.assertRaises(WebVisitError):
                store.visit('file:///tmp/private')
            self.assertEqual('https://example.test/', normalize_url('HTTPS://EXAMPLE.TEST:443'))

    def test_document_visit_cache_reuses_fixed_corpus_entry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_path = Path(temp_dir) / 'corpus.jsonl'
            corpus_path.write_text(
                json.dumps({'id': 'doc-1', 'url': 'https://example.test/1', 'text': 'full text'}) + '\n',
                encoding='utf-8',
            )
            store = CorpusDocumentStore(corpus_path)

            first = store.visit('doc-1')
            second = store.visit('doc-1')

            self.assertEqual(first, second)
            self.assertEqual(1, store.cache_info().misses)
            self.assertEqual(1, store.cache_info().hits)

    def test_batch_search_returns_raw_top_k_chunks_in_query_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            index_path = temp_path / 'chunks.faiss'
            chunks_path = temp_path / 'chunks.jsonl'

            index = faiss.IndexFlatIP(2)
            index.add(np.asarray([
                [1.0, 0.0],
                [0.0, 1.0],
                [0.7, 0.7],
            ], dtype='float32'))
            faiss.write_index(index, str(index_path))
            chunks = [
                {'doc_id': 'doc-a', 'chunk_id': 'a-0', 'text': 'first'},
                {'doc_id': 'doc-b', 'chunk_id': 'b-0', 'text': 'second'},
                {'doc_id': 'doc-c', 'chunk_id': 'c-0', 'text': 'shared'},
            ]
            chunks_path.write_text(
                ''.join(json.dumps(chunk) + '\n' for chunk in chunks),
                encoding='utf-8',
            )

            retriever = LocalRetriever(
                index_path,
                chunks_path,
                embedder_name='fake',
                embedder=FakeBatchEmbedder(),
            )
            results = retriever.search_batch(['first', 'second'], [2, 1])

            self.assertEqual(['a-0', 'c-0'], [item['chunk_id'] for item in results[0]])
            self.assertEqual(['b-0'], [item['chunk_id'] for item in results[1]])

    def test_search_tool_accepts_structured_batch(self):
        class FakeRetriever:
            def search_batch(self, queries, top_ks):
                self.call = (queries, top_ks)
                return [[{
                    'doc_id': f'doc-{index}',
                    'chunk_id': f'chunk-{index}',
                    'score': 1.0,
                    'text': query,
                }] for index, query in enumerate(queries)]

        retriever = FakeRetriever()
        tool_result = SearchTool(retriever).execute(queries=[
            {'query': 'first', 'top_k': 3},
            {'query': 'second'},
        ])
        results = tool_result.data['results']

        self.assertTrue(tool_result.ok)
        self.assertEqual((['first', 'second'], [3, 5]), retriever.call)
        self.assertEqual(['first', 'second'], [item['query'] for item in results])
        self.assertEqual('chunk-1', results[1]['chunks'][0]['chunk_id'])


if __name__ == '__main__':
    unittest.main()
