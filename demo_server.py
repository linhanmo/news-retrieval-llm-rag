from __future__ import annotations

import argparse
import json
import logging
import threading
import time
import warnings
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from config import DEFAULT_TOP_K, build_timestamped_result_path, ensure_directories
from rag_pipeline import NewsRAGPipeline
from report_formatter import build_display_report


ROOT_DIR = Path(__file__).resolve().parent
PIPELINE_LOCK = threading.Lock()
QUERY_LOCK = threading.Lock()
PIPELINE: NewsRAGPipeline | None = None


def configure_quiet_output() -> None:
    warnings.filterwarnings("ignore")
    for logger_name in ("sentence_transformers", "transformers", "urllib3", "chromadb", "llama_cpp"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def get_pipeline() -> NewsRAGPipeline:
    global PIPELINE
    with PIPELINE_LOCK:
        if PIPELINE is None:
            PIPELINE = NewsRAGPipeline()
        return PIPELINE


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT_DIR), **kwargs)

    def log_message(self, format: str, *args) -> None:
        return

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._send_json({"status": "ok", "service": "demo_server"})
            return
        if parsed.path == "/":
            self.path = "/showcase.html"
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/query":
            self._send_json({"error": "未找到接口。"}, status=HTTPStatus.NOT_FOUND)
            return

        content_length = int(self.headers.get("Content-Length", "0") or 0)
        raw_body = self.rfile.read(content_length) if content_length else b"{}"

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"error": "请求体不是合法 JSON。"}, status=HTTPStatus.BAD_REQUEST)
            return

        query = str(payload.get("query", "")).strip()
        top_k = int(payload.get("top_k", DEFAULT_TOP_K) or DEFAULT_TOP_K)
        top_k = max(1, min(top_k, 10))

        if not query:
            self._send_json({"error": "查询内容不能为空。"}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            started_at = time.time()
            pipeline = get_pipeline()
            with QUERY_LOCK:
                raw_result = pipeline.run(query, top_k=top_k)
            report = build_display_report(raw_result)

            output_path = build_timestamped_result_path()
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2)

            self._send_json(
                {
                    "ok": True,
                    "report": report,
                    "saved_to": str(output_path),
                    "elapsed_seconds": round(time.time() - started_at, 2),
                }
            )
        except Exception as exc:  # pragma: no cover
            self._send_json({"error": f"查询失败：{exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动本地新闻问答演示服务。")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址。")
    parser.add_argument("--port", type=int, default=8765, help="监听端口。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    configure_quiet_output()
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"演示服务已启动: http://{args.host}:{args.port}/showcase.html")
    server.serve_forever()


if __name__ == "__main__":
    main()
