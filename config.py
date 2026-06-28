from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
CHROMA_DIR = ROOT_DIR / "chroma_db"
CACHE_DIR = ROOT_DIR / ".cache"
HF_HOME_DIR = CACHE_DIR / "huggingface"
TRANSFORMERS_CACHE_DIR = HF_HOME_DIR / "transformers"
SENTENCE_TRANSFORMERS_HOME_DIR = CACHE_DIR / "sentence_transformers"
PIP_CACHE_DIR = CACHE_DIR / "pip"
TEMP_DIR = CACHE_DIR / "tmp"

QWEN_HF_DIR = ROOT_DIR / "qwen3.5-9b"
LLAMA_CPP_DIR = ROOT_DIR / "llama.cpp"
CHRONOQA_JSON_PATH = ROOT_DIR / "ChronoQA" / "chronoqa.json"
CHRONOQA_JSONL_PATH = ROOT_DIR / "ChronoQA" / "chronoqa.jsonl"

NORMALIZED_CORPUS_PATH = DATA_DIR / "normalized_news.jsonl"

GGUF_F16_PATH = MODELS_DIR / "qwen3.5-9b-f16.gguf"
GGUF_Q4_PATH = MODELS_DIR / "qwen3.5-9b-Q4_K_M.gguf"

CHROMA_COLLECTION_NAME = "chronoqa_rag"
LOCAL_EMBEDDING_CACHE_DIR = ROOT_DIR / "bge-m3"

DEFAULT_TOP_K = 8
DEFAULT_MAX_TOKENS = 1536
DEFAULT_TEMPERATURE = 0.2
DEFAULT_CONTEXT_SIZE = 8192
DEFAULT_THREADS = max(1, (os.cpu_count() or 4) - 1)
DEFAULT_GPU_LAYERS = 0
DEFAULT_BATCH_SIZE = 32

SOURCE_MAPPING = {
    "sina.com.cn": {
        "source_name": "新浪新闻",
        "country": "中国",
        "region": "CN",
        "language": "zh",
    },
}

BIAS_TERM_GROUPS = [
    {
        "category": "武装主体称谓",
        "terms": ["恐怖分子", "武装分子", "武装人员", "极端分子", "抵抗组织"],
    },
    {
        "category": "军事行动定性",
        "terms": ["入侵", "特别军事行动", "反恐行动", "清剿", "袭击", "打击"],
    },
    {
        "category": "政治身份定性",
        "terms": ["政权", "政府", "当局", "临时政府", "傀儡政权"],
    },
    {
        "category": "平民伤亡定性",
        "terms": ["附带损害", "平民伤亡", "人道主义灾难", "种族灭绝", "大屠杀"],
    },
]

SYSTEM_PROMPT = """你是一个本地新闻分析助手，任务是基于检索到的新闻材料生成结构化中文分析。

严格遵守以下要求：
1. 先给出中立、克制、可核查的事件摘要。
2. 再按媒体来源分别总结报道侧重点、叙事框架和高频用词。
3. 不要把推测写成事实；如果材料不足，明确说明“证据不足”。
4. 输出必须为合法 JSON，且所有字段名都必须使用中文，不要输出 JSON 之外的任何解释。
5. 顶层字段包括：
   中立摘要
   媒体对比
   潜在意识形态词
   证据
6. 媒体对比 是数组，每项包含：
   来源
   国家
   侧重点
   用词特征
7. 潜在意识形态词 是数组，每项包含：
   词语
   类别
   原因
8. 证据 是数组，每项包含：
   来源
   链接
   摘录
"""


def ensure_directories() -> None:
    for path in (
        DATA_DIR,
        MODELS_DIR,
        CHROMA_DIR,
        CACHE_DIR,
        HF_HOME_DIR,
        TRANSFORMERS_CACHE_DIR,
        SENTENCE_TRANSFORMERS_HOME_DIR,
        PIP_CACHE_DIR,
        TEMP_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def resolve_local_embedding_model_dir() -> Path:
    refs_main = LOCAL_EMBEDDING_CACHE_DIR / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot_dir = LOCAL_EMBEDDING_CACHE_DIR / "snapshots" / revision
        if (snapshot_dir / "modules.json").exists():
            return snapshot_dir

    if (LOCAL_EMBEDDING_CACHE_DIR / "modules.json").exists():
        return LOCAL_EMBEDDING_CACHE_DIR

    raise FileNotFoundError(
        f"未找到本地嵌入模型，请检查目录是否完整: {LOCAL_EMBEDDING_CACHE_DIR}"
    )


EMBEDDING_MODEL_NAME = str(resolve_local_embedding_model_dir())


def apply_local_environment() -> None:
    ensure_directories()
    os.environ.setdefault("HF_HOME", str(HF_HOME_DIR))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HOME_DIR / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(TRANSFORMERS_CACHE_DIR))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(SENTENCE_TRANSFORMERS_HOME_DIR))
    os.environ.setdefault("PIP_CACHE_DIR", str(PIP_CACHE_DIR))
    os.environ.setdefault("TMP", str(TEMP_DIR))
    os.environ.setdefault("TEMP", str(TEMP_DIR))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def build_timestamped_result_path() -> Path:
    ensure_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DATA_DIR / f"result_{timestamp}.json"
