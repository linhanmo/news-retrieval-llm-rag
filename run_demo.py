from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings

from config import DEFAULT_TOP_K, build_timestamped_result_path, ensure_directories
from rag_pipeline import NewsRAGPipeline
from report_formatter import build_display_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 ChronoQA 本地 RAG Demo。")
    parser.add_argument("--query", required=True, help="新闻事件关键词，如：巴以冲突停火协议。")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="检索文档数。")
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="关闭默认的流式输出。",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出 JSON 文件路径，默认按时间戳命名。",
    )
    return parser.parse_args()


def print_markdown(report: dict) -> None:
    answer_card = report.get("回答卡片", {})

    print("\n# 直接回答\n")
    print(answer_card.get("直接回答") or report.get("直接回答") or report.get("中立摘要", ""))

    print("\n# 中立摘要\n")
    print(report.get("中立摘要", ""))

    key_points = report.get("关键信息", [])
    if key_points:
        print("\n# 关键信息\n")
        for item in key_points:
            print(f"- {item}")

    print("\n# 报告概览\n")
    overview = report.get("报告概览", {})
    print(
        "- 回答状态: {回答状态} | 来源数量: {来源数量} | 证据数量: {证据数量} | 风险词数量: {风险词数量} | 检索文档数: {检索文档数}".format(
            回答状态=overview.get("回答状态", "未知"),
            来源数量=overview.get("来源数量", 0),
            证据数量=overview.get("证据数量", 0),
            风险词数量=overview.get("风险词数量", 0),
            检索文档数=overview.get("检索文档数", 0),
        )
    )

    print("\n# 媒体对比\n")
    for item in report.get("媒体对比", []):
        print(f"- 来源: {item.get('来源', '')} ({item.get('国家', '')})")
        print(f"  侧重点: {item.get('侧重点', '')}")
        print(f"  用词特征: {item.get('用词特征', '')}")

    print("\n# 倾向提示\n")
    risk_terms = report.get("倾向提示", [])
    if not risk_terms:
        print("- 未发现明显高风险词。")
    else:
        for item in risk_terms:
            print(f"- {item.get('词语', '')} [{item.get('识别方式', '')}]: {item.get('原因', '')}")

    print("\n# 证据\n")
    for item in report.get("证据", []):
        print(f"- {item.get('来源', '')} | {item.get('链接', '')}")
        print(f"  摘录: {item.get('摘录', '')}")


def stream_to_stdout(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def configure_quiet_output() -> None:
    warnings.filterwarnings("ignore")
    for logger_name in ("sentence_transformers", "transformers", "urllib3", "chromadb", "llama_cpp"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def main() -> None:
    args = parse_args()
    configure_quiet_output()
    ensure_directories()
    output_path = args.output or str(build_timestamped_result_path())
    use_stream = not args.no_stream

    pipeline = NewsRAGPipeline()
    if use_stream:
        print("\n# 流式输出\n")
        result = pipeline.run(args.query, top_k=args.top_k, on_chunk=stream_to_stdout)
        print("\n")
    else:
        result = pipeline.run(args.query, top_k=args.top_k)

    chinese_result = build_display_report(result)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(chinese_result, handle, ensure_ascii=False, indent=2)

    print_markdown(chinese_result)
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
