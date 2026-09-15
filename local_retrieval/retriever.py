import json
import hashlib
import warnings
from pathlib import Path

import faiss
import numpy as np
from .data_artifacts import sha256_file, validate_chunk

class LocalRetriever:
    def __init__(self, index_path, chunks_path, embedder_name='qwen3-0.6b', manifest_path=None, embedder=None, require_integrity=False):
        self.index_path = Path(index_path)
        self.chunks_path = Path(chunks_path)
        self.embedder_name = embedder_name

        self.index = faiss.read_index(str(self.index_path))
        digest = hashlib.sha256()
        self.chunks = []
        with self.chunks_path.open('rb') as source:
            for line in source:
                digest.update(line)
                if line.strip():
                    chunk = json.loads(line)
                    validate_chunk(chunk)
                    self.chunks.append(chunk)
            
        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                f"Index/chunk mismatch: "
                f"{self.index.ntotal} vectors vs "
                f"{len(self.chunks)} chunks"
            )

        manifest_path = Path(manifest_path or f'{self.index_path}.json')
        self.manifest = None
        if manifest_path.exists():
            with manifest_path.open(encoding='utf-8') as file:
                self.manifest = json.load(file)
            if self.manifest.get('embedder') != embedder_name:
                raise ValueError(
                    f"Index/embedder mismatch: manifest uses {self.manifest.get('embedder')}, "
                    f"requested {embedder_name}"
                )
            if self.manifest.get('vector_count') != self.index.ntotal:
                raise ValueError(
                    f"Manifest/index mismatch: {self.manifest.get('vector_count')} vectors in manifest vs "
                    f"{self.index.ntotal} in index"
                )

        if self.manifest and self.manifest.get('format_version') == 2:
            if self.manifest.get('chunks_sha256') != digest.hexdigest():
                raise ValueError('Index/chunks content hash mismatch')
            if self.manifest.get('index_sha256') != sha256_file(self.index_path):
                raise ValueError('Index content hash mismatch')
        elif require_integrity:
            raise ValueError('Integrity manifest v2 required; rebuild index or explicitly allow legacy index')
        else:
            warnings.warn('Legacy index: content integrity is not verified', RuntimeWarning, stacklevel=2)

        if embedder is None:
            from .embedder import create_embedder
            embedder = create_embedder(embedder_name)
        self.embedder = embedder
        if self.manifest and self.manifest.get('model') not in (None, self.embedder.model_name):
            raise ValueError('Manifest model does not match the configured embedding model')
        if self.index.d != self.embedder.dimension:
            raise ValueError(
                f'Index/embedder dimension mismatch: {self.index.d} in index vs '
                f'{self.embedder.dimension} from {self.embedder.model_name}'
            )
        if self.manifest and self.manifest.get('dimension') != self.index.d:
            raise ValueError(
                f"Manifest/index dimension mismatch: {self.manifest.get('dimension')} in manifest vs "
                f"{self.index.d} in index"
            )
        
    @staticmethod
    def _as_numpy(embeddings):
        if hasattr(embeddings, 'detach'):
            embeddings = embeddings.detach()
        if hasattr(embeddings, 'cpu'):
            embeddings = embeddings.cpu()
        if hasattr(embeddings, 'numpy'):
            embeddings = embeddings.numpy()
        return np.asarray(embeddings, dtype='float32')

    def _encode_queries(self, queries):
        if hasattr(self.embedder, 'encode_queries'):
            return self._as_numpy(self.embedder.encode_queries(queries))
        rows = [self._as_numpy(self.embedder.encode_query(query)) for query in queries]
        return np.concatenate(rows, axis=0)

    def _format_results(self, scores, indices, top_k):
        results = []
        for score, idx in zip(scores[:top_k], indices[:top_k]):
            chunk = self.chunks[idx]
            results.append({
                "doc_id": chunk["doc_id"],
                "chunk_id": chunk["chunk_id"],
                "score": float(score),
                "text": chunk["text"],
            })
        return results

    def search_batch(self, queries: list[str], top_ks: list[int] | None = None) -> list[list[dict]]:
        if not queries:
            return []
        top_ks = top_ks or [5] * len(queries)
        if len(top_ks) != len(queries):
            raise ValueError('top_ks must contain one value per query')
        if any(top_k <= 0 for top_k in top_ks):
            raise ValueError('top_k must be greater than zero')

        embeddings = self._encode_queries(queries)
        if embeddings.shape != (len(queries), self.index.d):
            raise ValueError(
                f'Query embedding shape mismatch: {embeddings.shape}, '
                f'expected {(len(queries), self.index.d)}'
            )
        search_k = min(max(top_ks), self.index.ntotal)
        scores, indices = self.index.search(embeddings, k=search_k)
        return [
            self._format_results(row_scores, row_indices, min(top_k, search_k))
            for row_scores, row_indices, top_k in zip(scores, indices, top_ks)
        ]

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        return self.search_batch([query], [top_k])[0]
