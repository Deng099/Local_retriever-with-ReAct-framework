"""Convert downloaded corpus Parquet shards to JSONL without loading all text."""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from .data_artifacts import artifact_pair


def convert_corpus(input_dir, output, batch_size=256, expected_docs=None):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    output = Path(output)
    shards = sorted(Path(input_dir).rglob('*.parquet'))
    if not shards:
        raise ValueError('No Parquet shards found')
    seen = set()
    stats = {'shards': len(shards), 'rows': 0, 'documents': 0, 'empty_text': 0}
    with (
        artifact_pair(output, Path(f'{output}.json'), inputs=shards) as (temporary, temporary_metadata),
        temporary.open('w', encoding='utf-8') as target,
    ):
        for shard in shards:
            parquet = pq.ParquetFile(shard)
            names = parquet.schema_arrow.names
            id_column = 'id' if 'id' in names else 'docid'
            if id_column not in names or 'text' not in names:
                raise ValueError(f'{shard}: expected id/docid and text columns, got {names}')
            columns = [id_column, 'text'] + [name for name in ('url', 'title') if name in names]
            for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
                for row in batch.to_pylist():
                    stats['rows'] += 1
                    doc_id = row[id_column]
                    if doc_id is None or not str(doc_id).strip():
                        raise ValueError(f'{shard}: missing document ID')
                    doc_id = str(doc_id)
                    if doc_id in seen:
                        raise ValueError(f'Duplicate document ID: {doc_id}')
                    seen.add(doc_id)
                    text = row['text']
                    if text is None or (isinstance(text, str) and not text.strip()):
                        stats['empty_text'] += 1
                        continue
                    if not isinstance(text, str):
                        raise ValueError(f'{doc_id}: text must be a string')
                    record = {'id': doc_id, 'text': text}
                    record.update({name: row[name] for name in ('url', 'title') if row.get(name) is not None})
                    target.write(json.dumps(record, ensure_ascii=False) + '\n')
                    stats['documents'] += 1
            print(f'read {shard.name}: {stats["rows"]} rows, {stats["documents"]} documents', flush=True)
        if expected_docs is not None and stats['documents'] != expected_docs:
            raise ValueError(f'Expected {expected_docs} documents, got {stats["documents"]}')
        if not stats['documents']:
            raise ValueError('Corpus has no nonempty documents')
        temporary_metadata.write_text(json.dumps(stats, indent=2), encoding='utf-8')
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--expected-docs', type=int)
    args = parser.parse_args()
    print(json.dumps(convert_corpus(args.input_dir, args.output, args.batch_size, args.expected_docs)))


if __name__ == '__main__':
    main()
