from __future__ import annotations

import json
import re
from collections import defaultdict
from json import JSONDecodeError
from typing import Any, Callable

import chromadb
from llama_cpp import Llama

from config import (
    BIAS_TERM_GROUPS,
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_GPU_LAYERS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_THREADS,
    EMBEDDING_MODEL_NAME,
    GGUF_Q4_PATH,
    SYSTEM_PROMPT,
    apply_local_environment,
)


class NewsRAGPipeline:
    def __init__(
        self,
        chroma_dir: str = str(CHROMA_DIR),
        collection_name: str = CHROMA_COLLECTION_NAME,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        model_path: str = str(GGUF_Q4_PATH),
        n_ctx: int = DEFAULT_CONTEXT_SIZE,
        n_threads: int = DEFAULT_THREADS,
        n_gpu_layers: int = DEFAULT_GPU_LAYERS,
    ) -> None:
        apply_local_environment()
        from sentence_transformers import SentenceTransformer

        self.embedder = SentenceTransformer(embedding_model_name, local_files_only=True)
        client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = client.get_collection(collection_name)
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def retrieve(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        query_embedding = self.embedder.encode([query], normalize_embeddings=True).tolist()[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        rows: list[dict[str, Any]] = []
        for index, metadata in enumerate(result["metadatas"][0]):
            rows.append(
                {
                    "document": result["documents"][0][index],
                    "metadata": metadata,
                    "distance": result["distances"][0][index],
                }
            )
        return rows

    def build_prompt(self, query: str, retrieved_docs: list[dict[str, Any]]) -> str:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in retrieved_docs:
            source_name = item["metadata"].get("source_name", "未知来源")
            grouped[source_name].append(item)

        evidence_blocks = []
        for source_name, items in grouped.items():
            for item in items:
                meta = item["metadata"]
                evidence_blocks.append(
                    "\n".join(
                        [
                            f"来源: {source_name}",
                            f"国家: {meta.get('country', '未知')}",
                            f"发布时间: {meta.get('publish_date', '未知')}",
                            f"链接: {meta.get('url', '')}",
                            f"标题: {meta.get('title', '')}",
                            f"内容: {item['document']}",
                        ]
                    )
                )

        return (
            f"用户查询: {query}\n\n"
            "以下是检索到的新闻证据，请仅基于这些材料作答:\n\n"
            + "\n\n---\n\n".join(evidence_blocks)
        )

    def generate_raw_text(self, prompt: str, on_chunk: Callable[[str], None] | None = None) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        if on_chunk is None:
            completion = self.llm.create_chat_completion(
                messages=messages,
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=DEFAULT_MAX_TOKENS,
                response_format={"type": "json_object"},
            )
            return completion["choices"][0]["message"]["content"]

        stream = self.llm.create_chat_completion(
            messages=messages,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
            response_format={"type": "json_object"},
            stream=True,
        )

        chunks: list[str] = []
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                chunks.append(content)
                on_chunk(content)

        return "".join(chunks)

    def generate(
        self,
        query: str,
        retrieved_docs: list[dict[str, Any]],
        on_chunk: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        prompt = self.build_prompt(query, retrieved_docs)
        raw_text = self.generate_raw_text(prompt, on_chunk=on_chunk)
        parsed = self.normalize_model_output(self.safe_parse_json(raw_text))
        parsed["retrieved_sources"] = self.summarize_sources(retrieved_docs)
        parsed["lexicon_matches"] = self.detect_bias_terms(json.dumps(parsed, ensure_ascii=False))
        return parsed

    def run(
        self,
        query: str,
        top_k: int = 8,
        on_chunk: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        retrieved_docs = self.retrieve(query, top_k=top_k)
        result = self.generate(query, retrieved_docs, on_chunk=on_chunk)
        result["query"] = query
        result["retrieved_documents"] = [
            {
                "source_name": item["metadata"].get("source_name", ""),
                "country": item["metadata"].get("country", ""),
                "url": item["metadata"].get("url", ""),
                "title": item["metadata"].get("title", ""),
                "publish_date": item["metadata"].get("publish_date", ""),
                "document": item["document"],
            }
            for item in retrieved_docs
        ]
        return result

    @staticmethod
    def summarize_sources(retrieved_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for item in retrieved_docs:
            metadata = item["metadata"]
            key = (metadata.get("source_name", ""), metadata.get("country", ""))
            grouped.setdefault(
                key,
                {
                    "source_name": metadata.get("source_name", ""),
                    "country": metadata.get("country", ""),
                    "count": 0,
                    "sample_titles": [],
                },
            )
            grouped[key]["count"] += 1
            title = metadata.get("title", "")
            if title and title not in grouped[key]["sample_titles"]:
                grouped[key]["sample_titles"].append(title)
        return list(grouped.values())

    @staticmethod
    def safe_parse_json(raw_text: str) -> dict[str, Any]:
        candidates: list[str] = []

        def add_candidate(value: str) -> None:
            normalized = value.strip()
            if normalized and normalized not in candidates:
                candidates.append(normalized)

        add_candidate(raw_text)
        add_candidate(NewsRAGPipeline._strip_code_fences(raw_text))

        extracted = NewsRAGPipeline._extract_outer_json_object(raw_text)
        if extracted:
            add_candidate(extracted)
            add_candidate(NewsRAGPipeline._strip_code_fences(extracted))

        for candidate in list(candidates):
            add_candidate(NewsRAGPipeline._remove_trailing_commas(candidate))

        last_error: JSONDecodeError | None = None
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except JSONDecodeError as exc:
                last_error = exc

        return NewsRAGPipeline._build_fallback_output(raw_text, last_error)

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        normalized = text.strip()
        normalized = re.sub(r"^```(?:json)?\s*", "", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\s*```$", "", normalized)
        return normalized.strip()

    @staticmethod
    def _extract_outer_json_object(text: str) -> str:
        start = text.find("{")
        if start == -1:
            return ""

        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        return ""

    @staticmethod
    def _remove_trailing_commas(text: str) -> str:
        return re.sub(r",\s*([}\]])", r"\1", text)

    @staticmethod
    def _extract_summary(raw_text: str) -> str:
        patterns = (
            r'"中立摘要"\s*:\s*"((?:[^"\\]|\\.)*)"',
            r'"neutral_summary"\s*:\s*"((?:[^"\\]|\\.)*)"',
        )
        for pattern in patterns:
            match = re.search(pattern, raw_text, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(f'"{match.group(1)}"')
                except JSONDecodeError:
                    return match.group(1).replace('\\"', '"').replace("\\n", "\n").strip()

        cleaned = NewsRAGPipeline._strip_code_fences(raw_text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return "模型输出解析失败，且未提取到可用摘要。"
        return cleaned[:500]

    @staticmethod
    def _build_fallback_output(raw_text: str, error: JSONDecodeError | None) -> dict[str, Any]:
        summary = NewsRAGPipeline._extract_summary(raw_text)
        if error is not None:
            summary = f"{summary}\n\n[解析提示] 模型返回了格式不完全合法的 JSON，已自动降级为文本摘要。"
        return {
            "中立摘要": summary,
            "媒体对比": [],
            "潜在意识形态词": [],
            "证据": [],
        }

    @staticmethod
    def normalize_model_output(parsed: dict[str, Any]) -> dict[str, Any]:
        def first_value(data: dict[str, Any], *keys: str, default: Any = "") -> Any:
            for key in keys:
                if key in data and data[key] is not None:
                    return data[key]
            return default

        source_comparison = []
        for item in first_value(parsed, "source_comparison", "媒体对比", default=[]):
            source_comparison.append(
                {
                    "source_name": first_value(item, "source_name", "来源"),
                    "country": first_value(item, "country", "国家"),
                    "focus": first_value(item, "focus", "侧重点"),
                    "wording_features": first_value(item, "wording_features", "用词特征"),
                }
            )

        ideology_risk_terms = []
        for item in first_value(parsed, "ideology_risk_terms", "潜在意识形态词", default=[]):
            ideology_risk_terms.append(
                {
                    "term": first_value(item, "term", "词语"),
                    "category": first_value(item, "category", "类别"),
                    "reason": first_value(item, "reason", "原因"),
                }
            )

        evidence = []
        for item in first_value(parsed, "evidence", "证据", default=[]):
            evidence.append(
                {
                    "source_name": first_value(item, "source_name", "来源"),
                    "url": first_value(item, "url", "链接"),
                    "excerpt": first_value(item, "excerpt", "摘录"),
                }
            )

        return {
            "neutral_summary": first_value(parsed, "neutral_summary", "中立摘要"),
            "source_comparison": source_comparison,
            "ideology_risk_terms": ideology_risk_terms,
            "evidence": evidence,
        }

    @staticmethod
    def detect_bias_terms(text: str) -> list[dict[str, str]]:
        matches: list[dict[str, str]] = []
        lowered = text.lower()
        for group in BIAS_TERM_GROUPS:
            for term in group["terms"]:
                if term.lower() in lowered:
                    matches.append(
                        {
                            "term": term,
                            "category": group["category"],
                            "reason": f"该词可能体现 {group['category']} 上的叙事倾向差异。",
                        }
                    )
        return matches
