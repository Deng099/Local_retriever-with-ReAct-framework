import argparse
import hashlib
import hmac
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from .document_store import CorpusDocumentStore, DEFAULT_INLINE_CONTENT_LIMIT
from .embedder import QWEN_NAME, create_embedder
from .retriever import LocalRetriever
from .web_document_store import WebDocumentStore, WebVisitError
from .visit_extraction import (
    DEFAULT_MAX_INPUT_CHARS,
    DEFAULT_MAX_OUTPUT_CHARS,
    DEFAULT_PROMPT_VERSION,
    OpenAIVisitExtractor,
    VisitExtractionCache,
)


load_dotenv()


def env_flag(name, default=True):
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {'1', 'true', 'yes', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'off'}:
        return False
    raise ValueError(f'{name} must be a boolean value')


def env_positive_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f'{name} must be a positive integer') from exc
    if parsed < 1:
        raise ValueError(f'{name} must be a positive integer')
    return parsed


class SearchItem(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=1000)


class SearchRequest(BaseModel):
    queries: list[SearchItem] = Field(min_length=1)


class VisitRequest(BaseModel):
    reference: str = Field(min_length=1)
    goal: str | None = Field(default=None, min_length=1)
    offset: int = Field(default=0, ge=0)
    limit: int | None = Field(default=None, ge=1)


def create_app(retriever, document_store=None, identity=None, api_key=None):
    app = FastAPI(title='Local retrieval service')

    def authorize(authorization):
        if not api_key:
            return
        scheme, separator, credential = (authorization or '').partition(' ')
        if not separator or scheme.lower() != 'bearer' or not hmac.compare_digest(credential, api_key):
            raise HTTPException(status_code=401, detail='Invalid retrieval service token')

    @app.get('/health')
    def health(authorization: str | None = Header(default=None)):
        authorize(authorization)
        return {
            'status': 'ok',
            'vectors': retriever.index.ntotal,
            'dimension': retriever.index.d,
            'embedder': retriever.embedder_name,
            'documents': len(document_store.documents) if document_store else 0,
            'identity': identity,
        }

    @app.post('/search')
    def search(request: SearchRequest, authorization: str | None = Header(default=None)):
        authorize(authorization)
        queries = [item.query for item in request.queries]
        top_ks = [item.top_k for item in request.queries]
        batch_results = retriever.search_batch(queries, top_ks)
        return {
            'results': [
                {'query': query, 'chunks': chunks}
                for query, chunks in zip(queries, batch_results)
            ]
        }

    @app.post('/visit')
    def visit(request: VisitRequest, authorization: str | None = Header(default=None)):
        authorize(authorization)
        if document_store is None:
            raise RuntimeError('Document store is not configured')
        try:
            return document_store.visit(
                request.reference,
                request.goal,
                offset=request.offset,
                limit=request.limit,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except WebVisitError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


def main():
    retrieval_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--index',
        type=Path,
        default=retrieval_dir / 'indexes/chunks_5k_qwen3_0.6b.faiss',
    )
    parser.add_argument(
        '--chunks',
        type=Path,
        default=retrieval_dir / 'data/chunks_5k.jsonl',
    )
    parser.add_argument(
        '--corpus',
        type=Path,
        default=retrieval_dir / 'data/corpus_5k.jsonl',
    )
    parser.add_argument(
        '--embedding-model',
        default=os.getenv('EMBEDDING_MODEL_PATH', 'Qwen/Qwen3-Embedding-0.6B'),
        help='Local model path or Hugging Face model ID; defaults to EMBEDDING_MODEL_PATH',
    )
    parser.add_argument(
        '--embedding-device',
        default=os.getenv('EMBEDDING_DEVICE'),
        help='Torch device such as cuda, cuda:0, or cpu; auto-detected when omitted',
    )
    parser.add_argument(
        '--visit-cache',
        type=Path,
        default=retrieval_dir / 'cache/visit.sqlite3',
    )
    parser.add_argument(
        '--inline-content-limit',
        type=int,
        default=DEFAULT_INLINE_CONTENT_LIMIT,
        help='Maximum document characters returned inline by visit; full text stays server-side',
    )
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--allow-legacy-index', action='store_true', help='Explicitly allow old indexes without content hashes')
    parser.add_argument('--port', type=int, default=8002)
    args = parser.parse_args()

    embedder = create_embedder(
        QWEN_NAME,
        model_path=args.embedding_model,
        device=args.embedding_device,
    )
    retriever = LocalRetriever(
        args.index,
        args.chunks,
        embedder_name=embedder.name,
        embedder=embedder,
        require_integrity=not args.allow_legacy_index,
    )
    # Extraction uses the generation configuration by default. Set
    # VISIT_EXTRACTION_MODEL/BASE_URL/API_KEY only when a separate extractor
    # is desired; this keeps the service portable across model providers.
    extraction_enabled = env_flag('VISIT_EXTRACTION_ENABLED')
    extraction_model = (
        (os.getenv('VISIT_EXTRACTION_MODEL') or os.getenv('OPENAI_MODEL'))
        if extraction_enabled else None
    )
    goal_extractor = None
    if extraction_model:
        extraction_api_key = os.getenv('VISIT_EXTRACTION_API_KEY') or os.getenv('OPENAI_API_KEY')
        if not extraction_api_key:
            raise ValueError('VISIT_EXTRACTION_API_KEY or OPENAI_API_KEY is required when VISIT_EXTRACTION_MODEL is set')
        goal_extractor = OpenAIVisitExtractor(
            model=extraction_model,
            api_key=extraction_api_key,
            base_url=os.getenv('VISIT_EXTRACTION_BASE_URL', os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')),
            prompt_version=os.getenv('VISIT_EXTRACTION_PROMPT_VERSION', DEFAULT_PROMPT_VERSION),
            max_input_chars=env_positive_int(
                'VISIT_EXTRACTION_MAX_INPUT_CHARS', DEFAULT_MAX_INPUT_CHARS
            ),
            max_output_chars=env_positive_int(
                'VISIT_EXTRACTION_MAX_OUTPUT_CHARS', DEFAULT_MAX_OUTPUT_CHARS
            ),
            trust_env=env_flag(
                'VISIT_EXTRACTION_TRUST_ENV', env_flag('OPENAI_TRUST_ENV')
            ),
        )
    document_store = CorpusDocumentStore(
        args.corpus,
        web_store=WebDocumentStore(args.visit_cache),
        goal_extractor=goal_extractor,
        extraction_cache=VisitExtractionCache(args.visit_cache),
        inline_content_limit=args.inline_content_limit,
    )
    identity = {
        'model': embedder.model_name,
        'model_source': str(embedder.model_path),
        'device': str(embedder.device),
        'query_instruction': embedder.query_instruction,
        'visit_extraction': {
            'enabled': goal_extractor is not None,
            'model': goal_extractor.model if goal_extractor else None,
            'prompt_version': goal_extractor.prompt_version if goal_extractor else None,
            'max_input_chars': goal_extractor.max_input_chars if goal_extractor else None,
            'max_output_chars': goal_extractor.max_output_chars if goal_extractor else None,
            'config_source': (
                'VISIT_EXTRACTION_MODEL'
                if goal_extractor and os.getenv('VISIT_EXTRACTION_MODEL')
                else ('OPENAI_MODEL' if goal_extractor else None)
            ),
            'inline_content_limit': args.inline_content_limit,
        },
        'files': {},
    }
    for name, path in [('index', args.index), ('chunks', args.chunks), ('corpus', args.corpus)]:
        digest = hashlib.sha256()
        with path.open('rb') as file:
            for block in iter(lambda: file.read(1024 * 1024), b''):
                digest.update(block)
        identity['files'][name] = digest.hexdigest()
    uvicorn.run(
        create_app(
            retriever,
            document_store,
            identity,
            api_key=os.getenv('RETRIEVER_API_KEY'),
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == '__main__':
    main()
