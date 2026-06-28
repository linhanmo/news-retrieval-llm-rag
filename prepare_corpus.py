from __future__ import annotations

import argparse
import json
import re
from typing import Any
from urllib.parse import urlparse

from tqdm import tqdm

from config import (
    CHRONOQA_JSONL_PATH,
    NORMALIZED_CORPUS_PATH,
    SOURCE_MAPPING,
    ensure_directories,
)


DATE_RE = re.compile(r"(\d{4}年\d{1,2}月\d{1,2}日)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 ChronoQA 规范化为新闻语料 JSONL。")
    parser.add_argument(
        "--input",
        default=str(CHRONOQA_JSONL_PATH),
        help="输入的 ChronoQA JSONL 文件路径。",
    )
    parser.add_argument(
        "--output",
        default=str(NORMALIZED_CORPUS_PATH),
        help="输出的标准化新闻 JSONL 文件路径。",
    )
    return parser.parse_args()


def infer_source(url: str) -> dict[str, str]:
    host = urlparse(url).netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    return {
        "source_name": SOURCE_MAPPING.get(host, {}).get("source_name", host or "unknown"),
        "country": SOURCE_MAPPING.get(host, {}).get("country", "未知"),
        "region": SOURCE_MAPPING.get(host, {}).get("region", "unknown"),
        "language": SOURCE_MAPPING.get(host, {}).get("language", "unknown"),
        "domain": host or "unknown",
    }


def clean_value(value: Any) -> Any:
    if isinstance(value, float) and str(value) == "nan":
        return None
    return value


def extract_date(text: str, fallback_date: str | None) -> str | None:
    match = DATE_RE.search(text)
    if match:
        return match.group(1)
    return fallback_date


def build_title(record: dict[str, Any], chunk_index: int, content: str) -> str:
    if record.get("event_name"):
        return str(record["event_name"])
    if record.get("original_question"):
        return str(record["original_question"])
    question = str(record.get("question") or "").strip()
    if question:
        return question
    return f"新闻片段 {chunk_index + 1}: {content[:32]}"


def normalize_record(record: dict[str, Any], row_index: int) -> list[dict[str, Any]]:
    golden_chunks = record.get("golden_chunks") or []
    golden_urls = record.get("golden_chunks_urls") or []
    documents: list[dict[str, Any]] = []

    for chunk_index, content in enumerate(golden_chunks):
        url = golden_urls[chunk_index] if chunk_index < len(golden_urls) else ""
        source_info = infer_source(url)
        content = str(content).strip()
        title = build_title(record, chunk_index, content)
        publish_date = extract_date(content, clean_value(record.get("event_time")) or clean_value(record.get("question_date")))
        topic = clean_value(record.get("event_name")) or clean_value(record.get("question"))

        documents.append(
            {
                "doc_id": f"chronoqa-{row_index:05d}-{chunk_index:02d}",
                "dataset": "ChronoQA",
                "title": title,
                "content": content,
                "topic": topic,
                "question": clean_value(record.get("question")),
                "answer": clean_value(record.get("answer")),
                "question_date": clean_value(record.get("question_date")),
                "publish_date": publish_date,
                "source_name": source_info["source_name"],
                "country": source_info["country"],
                "region": source_info["region"],
                "language": source_info["language"],
                "domain": source_info["domain"],
                "url": url,
                "temporal_type": clean_value(record.get("temporal_type")),
                "temporal_scope": clean_value(record.get("temporal_scope")),
                "answer_type": clean_value(record.get("answer_type")),
                "reference_document_count": clean_value(record.get("reference_document_count")),
            }
        )

    return documents


def main() -> None:
    args = parse_args()
    ensure_directories()

    input_path = CHRONOQA_JSONL_PATH.__class__(args.input)
    output_path = NORMALIZED_CORPUS_PATH.__class__(args.output)

    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc="读取 ChronoQA"):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    normalized_docs: list[dict[str, Any]] = []
    for row_index, row in enumerate(tqdm(rows, desc="规范化新闻文档")):
        normalized_docs.extend(normalize_record(row, row_index))

    with output_path.open("w", encoding="utf-8") as handle:
        for doc in tqdm(normalized_docs, desc="写入标准化语料"):
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"已写入 {len(normalized_docs)} 条新闻文档 -> {output_path}")


if __name__ == "__main__":
    main()
