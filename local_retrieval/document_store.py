import hashlib
import json
from functools import lru_cache
from pathlib import Path

from .web_document_store import WebVisitError, normalize_url


DEFAULT_INLINE_CONTENT_LIMIT = 12_000


class CorpusDocumentStore:
    '''In-memory lookup and visit cache for the fixed local corpus.'''

    def __init__(
        self,
        corpus_path,
        cache_size=1024,
        web_store=None,
        goal_extractor=None,
        extraction_cache=None,
        inline_content_limit=DEFAULT_INLINE_CONTENT_LIMIT,
    ):
        if isinstance(inline_content_limit, bool) or not isinstance(inline_content_limit, int):
            raise ValueError('inline_content_limit must be an integer')
        if inline_content_limit < 1:
            raise ValueError('inline_content_limit must be positive')
        self.corpus_path = Path(corpus_path)
        self.web_store = web_store
        self.goal_extractor = goal_extractor
        self.extraction_cache = extraction_cache
        self.inline_content_limit = inline_content_limit
        self.documents = {}
        with self.corpus_path.open(encoding='utf-8') as file:
            for line in file:
                if not line.strip():
                    continue
                source = json.loads(line)
                document = {
                    'doc_id': str(source['id']),
                    'url': source.get('url'),
                    'content': source['text'],
                    'content_hash': hashlib.sha256(source['text'].encode('utf-8')).hexdigest(),
                    'source': 'corpus',
                    'cache_hit': True,
                }
                self.documents[document['doc_id']] = document
                if document['url']:
                    self.documents[document['url']] = document
                    try:
                        self.documents[normalize_url(document['url'])] = document
                    except WebVisitError:
                        pass

        self._lookup = lru_cache(maxsize=cache_size)(self._lookup_uncached)

    def _lookup_uncached(self, reference):
        try:
            return self.documents[reference]
        except KeyError as exc:
            raise KeyError(f'Unknown corpus document or URL: {reference}') from exc

    def visit(self, reference, goal=None, offset=0, limit=None):
        reference = str(reference)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError('offset must be a non-negative integer')
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
        ):
            raise ValueError('limit must be a positive integer')
        try:
            document = dict(self._lookup(reference))
        except KeyError:
            if self.web_store is not None and reference.lower().startswith(('http://', 'https://')):
                normalized_reference = normalize_url(reference)
                try:
                    document = dict(self._lookup(normalized_reference))
                except KeyError:
                    document = self.web_store.visit(normalized_reference)
            else:
                raise
        if goal is None:
            return self._format_result(
                document,
                extraction=None,
                inline_content_limit=self.inline_content_limit,
                offset=offset,
                limit=limit,
            )
        from .visit_extraction import get_goal_extraction
        if self.goal_extractor is not None and self.extraction_cache is None:
            raise RuntimeError('Goal extraction cache is not configured')
        extraction = get_goal_extraction(
            document,
            goal,
            self.goal_extractor,
            self.extraction_cache,
        )
        return self._format_result(
            document,
            extraction,
            self.inline_content_limit,
            offset=offset,
            limit=limit,
        )

    @staticmethod
    def _format_result(
        document,
        extraction,
        inline_content_limit=DEFAULT_INLINE_CONTENT_LIMIT,
        offset=0,
        limit=None,
    ):
        content = document.get('content') or ''
        window_limit = limit or inline_content_limit
        inline_content = content[offset:offset + window_limit]
        cleaned_document = {
            key: document[key]
            for key in (
                'title',
                'content',
                'content_hash',
                'fetched_at',
                'status_code',
                'content_type',
                'parser_version',
            )
            if document.get(key) is not None
        }
        # Keep the full document in the retrieval service, but put only a
        # bounded preview in the tool result. This prevents one PDF/article
        # from consuming the agent's entire context window. The metadata lets
        # a caller decide whether to request another goal or a future range
        # read without silently treating the preview as the full document.
        cleaned_document['content'] = inline_content
        cleaned_document['content_length'] = len(content)
        cleaned_document['content_offset'] = offset
        cleaned_document['content_truncated'] = (
            offset > 0 or offset + len(inline_content) < len(content)
        )
        if cleaned_document['content_truncated']:
            cleaned_document['next_offset'] = offset + len(inline_content)
            if offset > 0:
                cleaned_document['previous_offset'] = max(0, offset - window_limit)
        if (
            extraction is not None
            and extraction.get('status') == 'ok'
            and cleaned_document['content_truncated']
        ):
            # For long goal-based visits the extraction is the model-facing
            # evidence view. Keep range metadata, but avoid duplicating even a
            # large preview beside it. A later visit without/with another goal
            # can request a different view from the server-side full text.
            cleaned_document['content'] = ''
            cleaned_document['content_omitted'] = True
        return {
            'source': {
                'type': document.get('source', 'corpus'),
                'doc_id': document.get('doc_id'),
                'url': document.get('url'),
                'normalized_url': document.get('normalized_url'),
            },
            'document': cleaned_document,
            'extraction': extraction,
            'cache': {
                'document_hit': bool(document.get('cache_hit')),
                'extraction_hit': (
                    extraction.get('cache_hit')
                    if extraction is not None
                    else None
                ),
            },
        }

    def cache_info(self):
        return self._lookup.cache_info()
