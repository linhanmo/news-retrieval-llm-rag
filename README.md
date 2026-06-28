# ChronoQA 本地 RAG

本项目当前仅使用 `ChronoQA` 数据集的本地 RAG 流程，目标能力包括：

- 针对新闻事件关键词进行检索
- 输出中立摘要
- 基于已检索新闻片段生成结构化摘要
- 标出可能具有意识形态倾向的敏感词

当前仓库只使用 `ChronoQA` ，需要注意：

- `ChronoQA` 主要是中文新闻片段集合
- 当前代码不再依赖外部新闻数据库
- 运行所需依赖、缓存、向量库与中间产物都落在项目目录内

## 目录说明

- `config.py`: 统一配置、路径和意识形态词表
- `prepare_model.ps1`: 构建 `llama.cpp`、转换 GGUF、量化到 `Q4_K_M`
- `prepare_corpus.py`: 将 `ChronoQA` 规范化成统一新闻语料
- `build_vector_store.py`: 构建 Chroma 向量库
- `rag_pipeline.py`: 检索 + 本地 LLM 生成主流程
- `report_formatter.py`: 将原始推理结果整理为页面友好的报告结构
- `run_demo.py`: 命令行入口
- `demo_server.py`: 本地演示服务
- `showcase.html`: 演示页面

## 1. 安装依赖并搭建环境

```powershell
conda create -n news python=3.9 -y
conda activate news
pip install -r requirements.txt
modelscope download --model Qwen/Qwen3.5-9B --local_dir ./qwen3.5-9b
```

## 2. 准备量化模型

把 `qwen3.5-9b` 转成 GGUF 并量化为 `Q4_K_M`，执行：

```powershell
.\prepare_model.ps1
```

完成后会生成：

- `models/qwen3.5-9b-f16.gguf`
- `models/qwen3.5-9b-Q4_K_M.gguf`

说明：

- `llama.cpp` 负责 `HF -> GGUF` 转换和 `Q4_K_M` 量化
- Python 侧脚本负责 RAG 与调用量化后的 GGUF 模型
- 转换与下载缓存默认写入项目内的 `.cache`

## 3. 规范化 ChronoQA

```powershell
python prepare_corpus.py
```

输出文件：

- `data/normalized_news.jsonl`

## 4. 构建 Chroma 向量库

```powershell
python build_vector_store.py --reset
```

默认使用本地离线嵌入模型：

- `bge-m3`

当前代码会强制离线：

- 不再访问 Hugging Face
- 只从本地 `bge-m3` 目录加载 embedding 模型
- 如果本地模型目录缺失或不完整，脚本会直接报错

## 5. 运行控制台 Demo

```powershell
python run_demo.py --query "2023诺贝尔奖得主"
```

输出结果会保存到：

- `data/result_时间戳.json`

当前保存的是整理后的中文报告结构，主要包含：

- `回答卡片`
- `直接回答`
- `中立摘要`
- `关键信息`
- `报告概览`
- `媒体对比`
- `倾向提示`
- `证据`
- `检索来源`
- `检索概览`

## 6. 启动演示页面

推荐直接启动本地演示服务：

```powershell
.\start_demo.ps1
```
也可以直接运行：

```powershell
python demo_server.py --port 8765
```

然后在浏览器打开：

- `http://127.0.0.1:8765/showcase.html`

健康检查地址：

- `http://127.0.0.1:8765/api/health`

页面会调用本地真实流程，并展示：

- 检索证据
- 结构化报告
- 媒体对比
- 原文摘录证据

## 本地目录约束

- Chroma 数据库存放在 `chroma_db`
- 标准化语料与结果文件写入 `data`
- 本地量化模型存放在 `models`
- 本地嵌入模型从 `bge-m3` 读取

## 当前范围

- 仅使用 `ChronoQA`
- 仅构建本地 Chroma 向量库
- 不引入额外数据库或外部新闻源
