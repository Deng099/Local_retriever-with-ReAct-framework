import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

from .retrieval_client import RetrievalClient


DEFAULT_CUTOFFS = (5, 100, 1000)


def load_queries(path):
    queries = []
    with path.open(newline='', encoding='utf-8') as file:
        for row in csv.reader(file, delimiter='\t'):
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                queries.append((row[0].strip(), row[1].strip()))
    return queries


def load_qrels(path):
    qrels = {}
    with path.open(encoding='utf-8') as file:
        for line in file:
            fields = line.split()
            if len(fields) != 4:
                continue
            query_id, _, doc_id, relevance = fields
            if int(relevance) > 0:
                qrels.setdefault(query_id, set()).add(str(doc_id))
    return qrels


def _first_ranked_docids(chunks):
    ranked_docids = []
    seen = set()
    for chunk in chunks:
        doc_id = str(chunk['doc_id'])
        if doc_id not in seen:
            seen.add(doc_id)
            ranked_docids.append(doc_id)
    return ranked_docids


def _recall_at(chunks, relevant_docids, cutoff):
    if not relevant_docids:
        return None
    retrieved = {str(chunk['doc_id']) for chunk in chunks[:cutoff]}
    return len(retrieved & relevant_docids) / len(relevant_docids)


def _ndcg_at(chunks, relevant_docids, cutoff):
    if not relevant_docids:
        return None
    seen = set()
    dcg = 0.0
    for rank, chunk in enumerate(chunks[:cutoff], start=1):
        doc_id = str(chunk['doc_id'])
        gain = doc_id in relevant_docids and doc_id not in seen
        seen.add(doc_id)
        if gain:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant_docids), cutoff)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_query(query_id, query, chunks, relevant_docids, cutoffs, elapsed_ms):
    relevant_docids = {str(doc_id) for doc_id in relevant_docids}
    ranked_docids = _first_ranked_docids(chunks)
    relevant_chunk_ranks = {}
    for rank, chunk in enumerate(chunks, start=1):
        doc_id = str(chunk['doc_id'])
        if doc_id in relevant_docids and doc_id not in relevant_chunk_ranks:
            relevant_chunk_ranks[doc_id] = rank
    metrics = {}
    for cutoff in cutoffs:
        metrics[f'recall@{cutoff}'] = _recall_at(chunks, relevant_docids, cutoff)
    metrics['ndcg@10'] = _ndcg_at(chunks, relevant_docids, 10)
    retrieved_relevant = relevant_docids & set(ranked_docids)
    return {
        'query_id': str(query_id),
        'query': query,
        'status': 'completed',
        'candidate_chunk_count': len(chunks),
        'unique_document_count': len(ranked_docids),
        'duplicate_chunk_count': len(chunks) - len(ranked_docids),
        'unique_documents_at_cutoff': {
            str(cutoff): len({str(chunk['doc_id']) for chunk in chunks[:cutoff]})
            for cutoff in cutoffs
        },
        'relevant_docids': sorted(relevant_docids),
        'retrieved_relevant_docids': sorted(retrieved_relevant),
        'missed_relevant_docids': sorted(relevant_docids - retrieved_relevant),
        'relevant_chunk_ranks': relevant_chunk_ranks,
        'ranked_docids': ranked_docids,
        'metrics': metrics,
        'latency_ms': round(elapsed_ms, 3),
    }


def summarize(results, cutoffs):
    completed = [result for result in results if result['status'] == 'completed']
    metric_names = [f'recall@{cutoff}' for cutoff in cutoffs] + ['ndcg@10']
    macro_metrics = {}
    for name in metric_names:
        values = [
            result['metrics'][name]
            for result in completed
            if result['metrics'][name] is not None
        ]
        macro_metrics[name] = sum(values) / len(values) if values else None
    return {
        'query_count': len(results),
        'completed_count': len(completed),
        'failed_count': len(results) - len(completed),
        'macro_metrics': macro_metrics,
        'mean_latency_ms': (
            sum(result['latency_ms'] for result in completed) / len(completed)
            if completed else None
        ),
        'mean_unique_documents': (
            sum(result['unique_document_count'] for result in completed) / len(completed)
            if completed else None
        ),
        'mean_duplicate_chunks': (
            sum(result['duplicate_chunk_count'] for result in completed) / len(completed)
            if completed else None
        ),
    }


def evaluate(retriever, queries, qrels, cutoffs=DEFAULT_CUTOFFS, batch_size=8):
    cutoffs = tuple(sorted(set(cutoffs)))
    if not cutoffs or cutoffs[0] < 1:
        raise ValueError('cutoffs must contain positive integers')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    candidate_k = max(cutoffs)
    results = []
    for start in range(0, len(queries), batch_size):
        batch = queries[start:start + batch_size]
        started = time.perf_counter()
        try:
            chunks_by_query = retriever.search_batch(
                [query for _, query in batch],
                [candidate_k] * len(batch),
            )
            elapsed_ms = (time.perf_counter() - started) * 1000 / len(batch)
            for (query_id, query), chunks in zip(batch, chunks_by_query):
                results.append(evaluate_query(
                    query_id,
                    query,
                    chunks,
                    qrels.get(str(query_id), set()),
                    cutoffs,
                    elapsed_ms,
                ))
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000 / len(batch)
            for query_id, query in batch:
                results.append({
                    'query_id': str(query_id),
                    'query': query,
                    'status': 'error',
                    'error': f'{type(exc).__name__}: {exc}',
                    'latency_ms': round(elapsed_ms, 3),
                })
    return {
        'metric_definition': (
            'Evidence-document recall within the first K chunk candidates. '
            'For nDCG, only the first chunk from each relevant document receives gain.'
        ),
        'cutoffs': list(cutoffs),
        'summary': summarize(results, cutoffs),
        'results': results,
    }


def save_report(report, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + '.tmp')
    with temporary_path.open('w', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    temporary_path.replace(output_path)


def main():
    project_root = Path(__file__).resolve().parent.parent
    default_data_dir = project_root / 'local_retrieval/data/bcp'
    parser = argparse.ArgumentParser(
        description='Evaluate the hosted chunk retriever without invoking an Agent or LLM.'
    )
    parser.add_argument(
        '--queries',
        type=Path,
        default=default_data_dir / 'queries_5k_all_evidence.tsv',
    )
    parser.add_argument(
        '--qrels',
        type=Path,
        default=default_data_dir / 'qrel_evidence.txt',
    )
    parser.add_argument('--output', type=Path, default=project_root / 'runs/retrieval_qwen_5k.json')
    parser.add_argument('--retriever-url', default='http://127.0.0.1:8002')
    parser.add_argument('--cutoffs', type=int, nargs='+', default=list(DEFAULT_CUTOFFS))
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()

    queries = load_queries(args.queries)
    if args.limit is not None:
        if args.limit < 1:
            parser.error('--limit must be positive')
        queries = queries[:args.limit]
    report = evaluate(
        RetrievalClient(args.retriever_url, api_key=os.getenv('RETRIEVER_API_KEY')),
        queries,
        load_qrels(args.qrels),
        cutoffs=args.cutoffs,
        batch_size=args.batch_size,
    )
    report['config'] = {
        'queries': str(args.queries),
        'qrels': str(args.qrels),
        'retriever_url': args.retriever_url,
        'batch_size': args.batch_size,
        'limit': args.limit,
    }
    save_report(report, args.output)
    print(json.dumps({'output': str(args.output), **report['summary']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
