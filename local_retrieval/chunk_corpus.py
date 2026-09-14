import json
from pathlib import Path
from transformers import AutoTokenizer

# ===== Config =====
RETRIEVAL_DIR = Path(__file__).resolve().parent
INPUT_PATH = RETRIEVAL_DIR / "data/corpus_5k.jsonl"
OUTPUT_PATH = RETRIEVAL_DIR / "data/chunks_5k.jsonl"

# Keep this tokenizer consistent with your embedding model.
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

CHUNK_SIZE = 256      # tokens per chunk
CHUNK_OVERLAP = 32    # overlapping tokens between adjacent chunks


def split_into_chunks( text: str, tokenizer, chunk_size: int = CHUNK_SIZE,
                        overlap: int = CHUNK_OVERLAP ) -> list[str]:
    
    """Split one document into token-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    if not token_ids:
        return []

    step = chunk_size - overlap
    chunks = []

    for start in range(0, len(token_ids), step):
        end = min(start + chunk_size, len(token_ids))
        chunk_ids = token_ids[start:end]

        chunk_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        if chunk_text:
            chunks.append(chunk_text)

        if end == len(token_ids):
            break

    return chunks


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    doc_count = 0
    chunk_count = 0

    with (
        INPUT_PATH.open("r", encoding="utf-8") as fin,
        OUTPUT_PATH.open("w", encoding="utf-8") as fout,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue

            doc = json.loads(line)

            # Compatible with either {"id": ..., "text": ...}
            # or {"docid": ..., "text": ...}.
            doc_id = doc.get("id", doc.get("docid"))
            text = doc.get("text", "")

            if doc_id is None:
                raise KeyError("Each document must contain 'id' or 'docid'")

            if not isinstance(text, str) or not text.strip():
                continue

            chunks = split_into_chunks(text, tokenizer)

            for chunk_id, chunk_text in enumerate(chunks):
                record = {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunk_count += 1

            doc_count += 1

    print(f"Processed documents: {doc_count}")
    print(f"Generated chunks: {chunk_count}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
