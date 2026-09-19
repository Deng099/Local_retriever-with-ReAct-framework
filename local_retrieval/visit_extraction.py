import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


# Bump when extraction input/output semantics change so old cache entries are
# not silently reused with a different evidence contract.
DEFAULT_PROMPT_VERSION = 'visit-goal-v2'
DEFAULT_MAX_INPUT_CHARS = 120_000
DEFAULT_MAX_OUTPUT_CHARS = 12_000


class VisitExtractionCache:
    def __init__(self, cache_path):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute('''
                CREATE TABLE IF NOT EXISTS visit_extractions (
                    content_hash TEXT NOT NULL,
                    goal_hash TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    extraction TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (content_hash, goal_hash, model, prompt_version)
                )
            ''')

    def _connect(self):
        return sqlite3.connect(self.cache_path, timeout=30)

    def get(self, content_hash, goal, model, prompt_version):
        goal_hash = self.goal_hash(goal)
        with self._connect() as connection:
            row = connection.execute('''
                SELECT extraction, created_at
                FROM visit_extractions
                WHERE content_hash = ? AND goal_hash = ?
                  AND model = ? AND prompt_version = ?
            ''', (content_hash, goal_hash, model, prompt_version)).fetchone()
        if row is None:
            return None
        return {'extraction': row[0], 'created_at': row[1]}

    def put(self, content_hash, goal, model, prompt_version, extraction):
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute('''
                INSERT INTO visit_extractions (
                    content_hash, goal_hash, model, prompt_version,
                    goal, extraction, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(content_hash, goal_hash, model, prompt_version)
                DO UPDATE SET
                    goal = excluded.goal,
                    extraction = excluded.extraction,
                    created_at = excluded.created_at
            ''', (
                content_hash,
                self.goal_hash(goal),
                model,
                prompt_version,
                goal,
                extraction,
                created_at,
            ))
        return created_at

    @staticmethod
    def goal_hash(goal):
        normalized_goal = ' '.join(goal.split())
        return hashlib.sha256(normalized_goal.encode('utf-8')).hexdigest()


class OpenAIVisitExtractor:
    '''Optional OpenAI-compatible goal extractor for already-cleaned documents.'''

    def __init__(
        self,
        model,
        api_key,
        base_url,
        prompt_version=DEFAULT_PROMPT_VERSION,
        max_input_chars=DEFAULT_MAX_INPUT_CHARS,
        max_output_chars=DEFAULT_MAX_OUTPUT_CHARS,
        trust_env=True,
        client=None,
    ):
        if client is None:
            from openai import DefaultHttpxClient, OpenAI
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                http_client=DefaultHttpxClient(trust_env=trust_env),
            )
        self.client = client
        self.model = model
        self.prompt_version = prompt_version
        if isinstance(max_input_chars, bool) or not isinstance(max_input_chars, int) or max_input_chars < 1:
            raise ValueError('max_input_chars must be a positive integer')
        self.max_input_chars = max_input_chars
        if isinstance(max_output_chars, bool) or not isinstance(max_output_chars, int) or max_output_chars < 1:
            raise ValueError('max_output_chars must be a positive integer')
        self.max_output_chars = max_output_chars
        self.last_metadata = {}

    def extract(self, document, goal):
        full_content = document['content']
        content = full_content[:self.max_input_chars]
        input_truncated = len(content) < len(full_content)
        source = document.get('url') or document.get('doc_id') or 'unknown source'
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    'role': 'system',
                    'content': (
                        'Extract only information from the supplied document that is relevant '
                        'to the stated goal. Preserve concrete names, dates, numbers, and '
                        'qualifications and quote short supporting spans when useful. Do not add '
                        'outside knowledge. If the document does not contain relevant information, '
                        'say so briefly. The supplied document may be truncated; do not infer that '
                        'a missing fact is absent from the complete document.'
                    ),
                },
                {
                    'role': 'user',
                    'content': (
                        f'Goal: {goal}\nSource: {source}\n'
                        f'Input characters: {len(content)} of {len(full_content)}\n\n'
                        f'Document excerpt:\n{content}'
                    ),
                },
            ],
        )
        extraction = response.choices[0].message.content
        if not extraction:
            raise RuntimeError('Visit extraction LM returned empty content')
        if len(extraction) > self.max_output_chars:
            extraction = (
                extraction[:self.max_output_chars]
                + '\n[Extraction truncated by the retrieval service.]'
            )
        self.last_metadata = {
            'input_chars': len(content),
            'input_truncated': input_truncated,
        }
        return extraction


def get_goal_extraction(document, goal, extractor, cache):
    goal = goal.strip()
    if not goal:
        return None

    if extractor is None:
        return {
            'status': 'not_configured',
            'goal': goal,
            'content': None,
            'model': None,
            'prompt_version': None,
            'created_at': None,
            'cache_hit': False,
        }

    content_hash = document.get('content_hash')
    if content_hash is None:
        content_hash = hashlib.sha256(document['content'].encode('utf-8')).hexdigest()
    cached = cache.get(
        content_hash,
        goal,
        extractor.model,
        extractor.prompt_version,
    )
    if cached is None:
        try:
            extraction_result = extractor.extract(document, goal)
        except Exception as exc:
            # Extraction is an optional context-reduction optimization. A
            # provider outage or unsupported model must not make the source
            # document unavailable to the agent; return a structured error
            # and let the document store keep its bounded raw preview.
            return {
                'status': 'error',
                'goal': goal,
                'content': None,
                'model': extractor.model,
                'prompt_version': extractor.prompt_version,
                'created_at': None,
                'cache_hit': False,
                'error': {
                    'type': type(exc).__name__,
                    'message': str(exc)[:2000],
                },
            }
        if isinstance(extraction_result, tuple):
            extraction, extraction_meta = extraction_result
        else:
            # Keep the small extractor protocol backwards compatible for
            # tests and downstream users that return only a string.
            extraction = extraction_result
            max_input_chars = getattr(extractor, 'max_input_chars', None)
            extraction_meta = getattr(extractor, 'last_metadata', None) or {
                'input_chars': min(len(document['content']), max_input_chars)
                if max_input_chars else len(document['content']),
                'input_truncated': bool(
                    max_input_chars and len(document['content']) > max_input_chars
                ),
            }
        created_at = cache.put(
            content_hash,
            goal,
            extractor.model,
            extractor.prompt_version,
            extraction,
        )
        cache_hit = False
    else:
        extraction = cached['extraction']
        created_at = cached['created_at']
        cache_hit = True
        max_input_chars = getattr(extractor, 'max_input_chars', None)
        extraction_meta = {
            'input_chars': min(len(document['content']), max_input_chars)
            if max_input_chars else len(document['content']),
            'input_truncated': bool(
                max_input_chars and len(document['content']) > max_input_chars
            ),
        }

    return {
        'status': 'ok',
        'goal': goal,
        'content': extraction,
        'model': extractor.model,
        'prompt_version': extractor.prompt_version,
        'created_at': created_at,
        'cache_hit': cache_hit,
        **extraction_meta,
    }
