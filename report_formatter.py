from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from config import SOURCE_MAPPING


def _clean_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return default
    return text


def _clean_document_text(value: Any) -> str:
    text = _clean_text(value)
    if text.lower().startswith("nan\n"):
        text = text[4:].strip()
    return text


def _first_sentence(text: str) -> str:
    normalized = _clean_text(text)
    if not normalized:
        return ""
    for separator in ("。", "！", "？", "\n"):
        if separator in normalized:
            return normalized.split(separator, 1)[0].strip() + ("。" if separator != "\n" else "")
    return normalized


def _split_sentences(text: str) -> list[str]:
    normalized = _clean_text(text)
    if not normalized:
        return []
    pieces = re.split(r"[。！？\n]+", normalized)
    return [piece.strip() for piece in pieces if piece.strip()]


def _build_key_points(summary: str, limit: int = 3) -> list[str]:
    points: list[str] = []
    for sentence in _split_sentences(summary):
        if sentence not in points:
            points.append(sentence)
        if len(points) >= limit:
            break
    return points


def _infer_answer_status(summary: str) -> str:
    normalized = _clean_text(summary)
    if not normalized:
        return "未生成"
    weak_signals = ("证据不足", "未明确提及", "并不完整", "无法确认", "未提及", "信息不足")
    if any(signal in normalized for signal in weak_signals):
        return "证据不足"
    return "已回答"


def _merge_text_parts(*parts: str) -> str:
    merged: list[str] = []
    for part in parts:
        text = _clean_text(part)
        if text and text not in merged:
            merged.append(text)
    return "；".join(merged)


def _resolve_source_display_name(source_name: Any, url: Any = None) -> str:
    normalized_source = _clean_text(source_name)
    normalized_url = _clean_text(url)

    candidates: list[str] = []
    if normalized_source:
        candidates.append(normalized_source)

    if normalized_url:
        parsed = urlparse(normalized_url)
        if parsed.netloc:
            candidates.append(parsed.netloc.lower())

    for candidate in candidates:
        mapping = SOURCE_MAPPING.get(candidate)
        if mapping:
            return mapping.get("source_name") or candidate

    return normalized_source or "未知来源"


def _resolve_country(source_name: Any, country: Any, url: Any = None) -> str:
    normalized_country = _clean_text(country)
    if normalized_country:
        return normalized_country

    normalized_source = _clean_text(source_name)
    normalized_url = _clean_text(url)
    for candidate in (normalized_source, urlparse(normalized_url).netloc.lower() if normalized_url else ""):
        mapping = SOURCE_MAPPING.get(candidate)
        if mapping and mapping.get("country"):
            return mapping["country"]
    return "未知"


def _derive_title(title: Any, document: Any) -> str:
    normalized_title = _clean_text(title)
    if normalized_title:
        return normalized_title

    document_text = _clean_document_text(document)
    if not document_text:
        return "未提供标题"

    first_line = document_text.splitlines()[0].strip()
    if first_line and len(first_line) <= 40:
        return first_line
    return _first_sentence(document_text)[:40] or "未提供标题"


def _dedupe_risk_terms(raw_result: dict[str, Any]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    groups = (
        ("模型识别", raw_result.get("ideology_risk_terms", [])),
        ("词表匹配", raw_result.get("lexicon_matches", [])),
    )

    for source, items in groups:
        for item in items:
            term = _clean_text(item.get("term") or item.get("词语"))
            category = _clean_text(item.get("category") or item.get("类别"), "未分类")
            reason = _clean_text(item.get("reason") or item.get("原因"))
            if not term:
                continue
            key = (term, category)
            if key in seen:
                continue
            seen.add(key)
            merged.append(
                {
                    "词语": term,
                    "类别": category,
                    "原因": reason,
                    "识别方式": source,
                }
            )
    return merged


def _build_media_comparison(raw_result: dict[str, Any]) -> list[dict[str, str]]:
    merged: dict[tuple[str, str], dict[str, str]] = {}

    for item in raw_result.get("source_comparison", []):
        source = _resolve_source_display_name(item.get("source_name"))
        country = _resolve_country(item.get("source_name"), item.get("country"))
        key = (source, country)
        current = merged.setdefault(
            key,
            {
                "来源": source,
                "国家": country,
                "侧重点": "",
                "用词特征": "",
            },
        )
        current["侧重点"] = _merge_text_parts(current["侧重点"], item.get("focus"))
        current["用词特征"] = _merge_text_parts(current["用词特征"], item.get("wording_features"))

    return list(merged.values())


def build_display_report(raw_result: dict[str, Any]) -> dict[str, Any]:
    summary = _clean_text(raw_result.get("neutral_summary"))
    direct_answer = _first_sentence(summary)
    answer_status = _infer_answer_status(summary)
    key_points = _build_key_points(summary)
    media_comparison = _build_media_comparison(raw_result)

    evidence_items = []
    for item in raw_result.get("evidence", []):
        evidence_items.append(
            {
                "来源": _resolve_source_display_name(item.get("source_name"), item.get("url")),
                "链接": _clean_text(item.get("url")),
                "摘录": _clean_text(item.get("excerpt")),
            }
        )

    retrieval_overview = []
    for item in raw_result.get("retrieved_documents", []):
        title = _derive_title(item.get("title"), item.get("document"))
        document = _clean_document_text(item.get("document"))
        source_name = _resolve_source_display_name(item.get("source_name"), item.get("url"))
        country = _resolve_country(item.get("source_name"), item.get("country"), item.get("url"))
        retrieval_overview.append(
            {
                "来源": source_name,
                "国家": country,
                "标题": title,
                "发布时间": _clean_text(item.get("publish_date"), "未知"),
                "链接": _clean_text(item.get("url")),
                "内容片段": document[:180] + ("..." if len(document) > 180 else ""),
            }
        )

    source_overview = []
    for item in raw_result.get("retrieved_sources", []):
        sample_titles = [
            title
            for title in (_clean_text(title) for title in item.get("sample_titles", []))
            if title
        ]
        source_overview.append(
            {
                "来源": _resolve_source_display_name(item.get("source_name")),
                "国家": _resolve_country(item.get("source_name"), item.get("country")),
                "条数": int(item.get("count", 0) or 0),
                "示例标题": sample_titles,
            }
        )

    risk_terms = _dedupe_risk_terms(raw_result)

    return {
        "查询": _clean_text(raw_result.get("query")),
        "回答卡片": {
            "回答状态": answer_status,
            "直接回答": direct_answer,
            "说明": summary if answer_status == "证据不足" else "",
        },
        "直接回答": direct_answer,
        "中立摘要": summary,
        "关键信息": key_points,
        "报告概览": {
            "回答状态": answer_status,
            "来源数量": len(source_overview),
            "证据数量": len(evidence_items),
            "风险词数量": len(risk_terms),
            "检索文档数": len(retrieval_overview),
        },
        "媒体对比": media_comparison,
        "倾向提示": risk_terms,
        "证据": evidence_items,
        "检索来源": source_overview,
        "检索概览": retrieval_overview,
    }
