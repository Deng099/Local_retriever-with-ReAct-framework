import argparse
import json
import hashlib
import os
from itertools import islice
from pathlib import Path

import faiss
import numpy as np

from .embedder import EMBEDDER_NAMES, create_embedder
from .data_artifacts import artifact_pair, check_output_paths, sha256_file, validate_chunk


DEFAULT_INDEX_NAMES = {
    'qwen3-0.6b': 'chunks_5k_qwen3_0.6b.faiss',
}


def load_chunks(path, digest=None):
    with Path(path).open('rb') as file:
        for line in file:
            if digest is not None:
                digest.update(line)
            if line.strip():
                chunk = json.loads(line)
                validate_chunk(chunk)
                yield chunk


def build_index(chunks, embedder, batch_size):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    index = faiss.IndexFlatIP(embedder.dimension)
    chunks = iter(chunks)
    batch_number = 0
    while True:
        rows = list(islice(chunks, batch_size))
        for chunk in rows:
            validate_chunk(chunk)
        batch = [chunk['text'] for chunk in rows]
        if not batch:
            break
        embeddings = embedder.encode_documents(batch)
        if hasattr(embeddings, 'detach'):
            embeddings = embeddings.detach().cpu().numpy()
        embeddings = np.asarray(embeddings, dtype='float32')
        if embeddings.shape != (len(batch), embedder.dimension) or not np.isfinite(embeddings).all():
            raise ValueError('Embedding batch has invalid shape or nonfinite values')
        index.add(embeddings)
        batch_number += 1
        if batch_number % 25 == 0:
            print(f'encoded: {index.ntotal} chunks', flush=True)
    if not index.ntotal:
        raise ValueError('No chunks to index')
    print(f'encoded: {index.ntotal} chunks', flush=True)
    return index


def save_manifest(path, embedder, chunks_path, index, chunk_size, chunk_overlap, chunks_sha256, index_sha256):
    chunks_path = Path(chunks_path)
    try:
        recorded_chunks_path = chunks_path.resolve().relative_to(Path(__file__).resolve().parent).as_posix()
    except ValueError:
        recorded_chunks_path = str(chunks_path)
    manifest = {
        'format_version': 2,
        'chunks_sha256': chunks_sha256,
        'index_sha256': index_sha256,
        'embedder': embedder.name,
        'model': embedder.model_name,
        'dimension': embedder.dimension,
        'pooling': embedder.pooling,
        'normalized': True,
        'query_instruction': embedder.query_instruction,
        'max_length': embedder.max_length,
        'chunks_path': recorded_chunks_path,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'vector_count': index.ntotal,
    }
    with path.open('w', encoding='utf-8') as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


def main():
    retrieval_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedder', choices=EMBEDDER_NAMES, default='qwen3-0.6b')
    parser.add_argument('--model-path', default=os.getenv('EMBEDDING_MODEL_PATH'), help='Local Qwen embedding model directory')
    parser.add_argument('--device', default=os.getenv('EMBEDDING_DEVICE'))
    parser.add_argument('--chunks', type=Path, default=retrieval_dir / 'data/chunks_5k.jsonl')
    parser.add_argument('--index', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--chunk-size', type=int, default=256)
    parser.add_argument('--chunk-overlap', type=int, default=32)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error('--batch-size must be greater than zero')
    if args.chunk_size <= 0 or not 0 <= args.chunk_overlap < args.chunk_size:
        parser.error('Require chunk_size > 0 and 0 <= chunk_overlap < chunk_size')

    index_path = args.index or retrieval_dir / 'indexes' / DEFAULT_INDEX_NAMES[args.embedder]
    manifest_path = args.manifest or Path(f'{index_path}.json')
    chunk_metadata = Path(f'{args.chunks}.json')
    check_output_paths(index_path, manifest_path, inputs=[args.chunks, chunk_metadata])
    metadata = None
    if chunk_metadata.exists():
        metadata = json.loads(chunk_metadata.read_text(encoding='utf-8'))
        if (metadata['chunk_size'], metadata['chunk_overlap']) != (args.chunk_size, args.chunk_overlap):
            parser.error('Chunk size/overlap do not match chunk metadata')
    digest = hashlib.sha256()
    chunks = load_chunks(args.chunks, digest=digest)
    embedder = create_embedder(args.embedder, model_path=args.model_path, device=args.device)
    index = build_index(chunks, embedder, args.batch_size)
    if metadata is not None and index.ntotal != metadata['chunks']:
        raise ValueError('Vector count does not match chunk metadata; no index published')
    if sha256_file(args.chunks) != digest.hexdigest():
        raise ValueError('Chunks changed during encoding; no index published')

    with artifact_pair(index_path, manifest_path, inputs=[args.chunks, chunk_metadata]) as (temporary_index, temporary_manifest):
        faiss.write_index(index, str(temporary_index))
        save_manifest(temporary_manifest, embedder, args.chunks, index, args.chunk_size, args.chunk_overlap,
                      digest.hexdigest(), sha256_file(temporary_index))
    print(f'vectors: {index.ntotal}')
    print(f'dimension: {index.d}')
    print(f'saved: {index_path}')
    print(f'manifest: {manifest_path}')


if __name__ == "__main__":
    main()
