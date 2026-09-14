import argparse
import json
from pathlib import Path

import faiss

from .embedder import EMBEDDER_NAMES, create_embedder


DEFAULT_INDEX_NAMES = {
    'qwen3-0.6b': 'chunks_5k_qwen3_0.6b.faiss',
}


def load_chunks(path):
    with path.open(encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]


def build_index(chunks, embedder, batch_size):
    index = faiss.IndexFlatIP(embedder.dimension)
    for start in range(0, len(chunks), batch_size):
        batch = [chunk['text'] for chunk in chunks[start:start + batch_size]]
        embeddings = embedder.encode_documents(batch).cpu().numpy().astype('float32')
        index.add(embeddings)
        end = min(start + batch_size, len(chunks))
        batch_number = start // batch_size + 1
        if batch_number % 25 == 0 or end == len(chunks):
            print(f'encoded: {end}/{len(chunks)}', flush=True)
    return index


def save_manifest(path, embedder, chunks_path, index, chunk_size, chunk_overlap):
    chunks_path = Path(chunks_path)
    try:
        recorded_chunks_path = chunks_path.resolve().relative_to(Path(__file__).resolve().parent).as_posix()
    except ValueError:
        recorded_chunks_path = str(chunks_path)
    manifest = {
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
    parser.add_argument('--chunks', type=Path, default=retrieval_dir / 'data/chunks_5k.jsonl')
    parser.add_argument('--index', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--chunk-size', type=int, default=256)
    parser.add_argument('--chunk-overlap', type=int, default=32)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error('--batch-size must be greater than zero')

    index_path = args.index or retrieval_dir / 'indexes' / DEFAULT_INDEX_NAMES[args.embedder]
    manifest_path = args.manifest or Path(f'{index_path}.json')
    chunks = load_chunks(args.chunks)
    embedder = create_embedder(args.embedder)
    index = build_index(chunks, embedder, args.batch_size)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    save_manifest(manifest_path, embedder, args.chunks, index, args.chunk_size, args.chunk_overlap)
    print(f'chunks: {len(chunks)}')
    print(f'vectors: {index.ntotal}')
    print(f'dimension: {index.d}')
    print(f'saved: {index_path}')
    print(f'manifest: {manifest_path}')


if __name__ == "__main__":
    main()
