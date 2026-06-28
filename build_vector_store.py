from __future__ import annotations

import argparse
import json
from pathlib import Path

import chromadb
from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    DEFAULT_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    NORMALIZED_CORPUS_PATH,
    apply_local_environment,
    ensure_directories,
)
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于本地 ChronoQA 语料构建离线 Chroma 向量库。")
    parser.add_argument("--input", default=str(NORMALIZED_CORPUS_PATH), help="标准化语料 JSONL 路径。")
    parser.add_argument("--db-dir", default=str(CHROMA_DIR), help="Chroma 持久化目录。")
    parser.add_argument("--collection", default=CHROMA_COLLECTION_NAME, help="向量集合名称。")
    parser.add_argument("--embed-model", default=EMBEDDING_MODEL_NAME, help="嵌入模型名称。")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="批量写入大小。")
    parser.add_argument("--reset", action="store_true", help="重建集合。")
    return parser.parse_args()


def load_documents(input_path: Path) -> list[dict]:
    documents: list[dict] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc="加载标准化语料"):
            line = line.strip()
            if not line:
                continue
            documents.append(json.loads(line))
    return documents


def batched(items: list[dict], batch_size: int) -> list[list[dict]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def main() -> None:
    args = parse_args()
    apply_local_environment()
    ensure_directories()
    from sentence_transformers import SentenceTransformer

    input_path = Path(args.input)
    db_dir = Path(args.db_dir)
    docs = load_documents(input_path)

    model = SentenceTransformer(args.embed_model, local_files_only=True)
    client = chromadb.PersistentClient(path=str(db_dir))

    if args.reset:
        try:
            client.delete_collection(args.collection)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={"description": "ChronoQA 离线 RAG 语料集合"},
    )

    for batch in tqdm(batched(docs, args.batch_size), desc="写入 Chroma"):
        texts = [f"{item['title']}\n{item['content']}" for item in batch]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
        ids = [item["doc_id"] for item in batch]
        metadatas = []
        for item in batch:
            metadatas.append(
                {
                    "dataset": str(item.get("dataset", "")),
                    "title": str(item.get("title", "")),
                    "topic": str(item.get("topic", "")),
                    "source_name": str(item.get("source_name", "")),
                    "country": str(item.get("country", "")),
                    "region": str(item.get("region", "")),
                    "language": str(item.get("language", "")),
                    "domain": str(item.get("domain", "")),
                    "url": str(item.get("url", "")),
                    "publish_date": str(item.get("publish_date", "")),
                    "question": str(item.get("question", "")),
                    "answer": str(item.get("answer", "")),
                    "temporal_type": str(item.get("temporal_type", "")),
                    "temporal_scope": str(item.get("temporal_scope", "")),
                }
            )
        collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    print(f"向量库构建完成: {args.collection} -> {db_dir}")


if __name__ == "__main__":
    main()
