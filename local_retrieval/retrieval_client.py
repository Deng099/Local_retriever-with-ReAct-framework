import json
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


class RetrievalClient:
    '''Small client for the long-running retrieval HTTP service.'''

    def __init__(self, base_url='http://127.0.0.1:8002', timeout=120, api_key=None):
        self.base_url = base_url.rstrip('/')
        self.endpoint = f'{self.base_url}/search'
        self.timeout = timeout
        self.api_key = api_key
        self.opener = build_opener(ProxyHandler({}))

    def _headers(self):
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers

    def search_batch(self, queries, top_ks=None):
        '''
        每个query设置top_k, 默认5
        queries=["What is ReAct?", "What is RAG?"],
        top_ks=[3, 5]
        '''
        if not queries:
            return []
        top_ks = top_ks or [5] * len(queries)
        # 万一填漏
        if len(top_ks) != len(queries):
            raise ValueError('top_ks must contain one value per query')
        requests = [
            {'query': query, 'top_k': top_k}
            for query, top_k in zip(queries, top_ks)
        ]
        # 转成 HTTP 能发送的数据
        payload = json.dumps({'queries': requests}).encode('utf-8')
        request = Request(
            self.endpoint,
            data=payload,
            headers=self._headers(),
            method='POST',
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                body = json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='replace')
            raise RuntimeError(f'Retrieval request failed with HTTP {exc.code}: {detail}') from exc
        except URLError as exc:
            raise RuntimeError(f'Cannot reach retrieval service at {self.endpoint}: {exc.reason}') from exc

        results = body.get('results')
        if not isinstance(results, list) or len(results) != len(requests):
            raise ValueError('Retrieval service returned an invalid batch response')
        return [result['chunks'] for result in results]

    def search(self, query, top_k=5):
        return self.search_batch([query], [top_k])[0]

    def health(self):
        request = Request(f'{self.base_url}/health', headers=self._headers())
        with self.opener.open(request, timeout=self.timeout) as response:
            body = json.load(response)
        if body.get('status') != 'ok':
            raise ValueError('Retrieval service is not healthy')
        return body

    def visit(self, reference, goal=None, offset=0, limit=None):
        arguments = {'reference': reference}
        if goal is not None:
            arguments['goal'] = goal
        if offset:
            arguments['offset'] = offset
        if limit is not None:
            arguments['limit'] = limit
        payload = json.dumps(arguments).encode('utf-8')
        request = Request(
            f'{self.base_url}/visit',
            data=payload,
            headers=self._headers(),
            method='POST',
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='replace')
            raise RuntimeError(f'Visit request failed with HTTP {exc.code}: {detail}') from exc
        except URLError as exc:
            raise RuntimeError(f'Cannot reach retrieval service at {self.base_url}: {exc.reason}') from exc
