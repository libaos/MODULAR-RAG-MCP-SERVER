# Modular RAG Repro

这个目录是对原项目的“功能等价复现版”。

目标不是逐文件照抄，而是按阶段重新实现一套最小可用的：

- 文档摄取
- 混合检索
- MCP 工具服务
- Dashboard
- Trace 与评估
- 文档生命周期管理

---

## 1. 这份复现版现在实现了什么

当前已经完成到：

- `Phase 0` 到 `Phase 8`
- 其中 `Phase 8` 的测试已经补齐

现在已经具备这些能力：

- PDF 摄取
- SHA256 去重
- 文本分块
- ChunkRefiner（规则版）
- MetadataEnricher（规则版）
- 图片落盘与本地索引
- Ollama Embedding
- BM25 稀疏索引
- Chroma 向量入库
- Dense 检索
- Sparse 检索
- RRF 融合
- 本地 Simple Rerank
- CLI 查询
- MCP Server
- Dashboard 六页
- ingest/query trace
- Golden Set 评估
- 文档摘要
- 文档删除
- BM25 重建钩子

---

## 2. 当前目录结构

```text
repro/
  config/
  data/
  logs/
  scripts/
  src/
    modular_rag_repro/
  tests/
    unit/
    integration/
    e2e/
```

主要入口：

- `scripts/ingest.py`
- `scripts/query.py`
- `scripts/evaluate.py`
- `scripts/mcp_server.py`
- `scripts/start_dashboard.py`

---

## 3. 当前运行前提

当前这份复现版还没有单独拆出自己的 `.venv`，默认复用外层项目的虚拟环境：

```powershell
cd "D:\OneDrive - stu.scau.edu.cn\桌面\workspace\MODULAR-RAG-MCP-SERVER\repro"
..\.venv\Scripts\Activate.ps1
```

当前默认配置是本地优先：

- LLM: `ollama / qwen2.5:3b`
- Embedding: `ollama / nomic-embed-text`
- Vector Store: `chroma`
- BM25: 本地 JSON 索引
- Rerank: `simple`（默认关闭，打开 `enabled` 即可生效）

建议先确认 Ollama 可用：

```powershell
ollama list
```

至少应有：

- `qwen2.5:3b`
- `nomic-embed-text`

---

## 4. 快速开始

### 4.1 摄取一个样例 PDF

```powershell
..\.venv\Scripts\python.exe scripts\ingest.py --path ..\tests\fixtures\sample_documents\simple.pdf --collection demo-repro
```

---

### 4.2 查询这个集合

```powershell
..\.venv\Scripts\python.exe scripts\query.py --query "sample pdf" --collection demo-repro --top-k 3 --verbose
```

---

### 4.3 跑 Golden Set 评估

```powershell
..\.venv\Scripts\python.exe scripts\evaluate.py --test-set tests\fixtures\golden_test_set.json --collection demo-repro --top-k 3
```

---

### 4.4 启动 MCP Server

```powershell
..\.venv\Scripts\python.exe scripts\mcp_server.py
```

---

### 4.5 启动 Dashboard

```powershell
..\.venv\Scripts\python.exe scripts\start_dashboard.py --port 8502
```

打开：

- `http://localhost:8502`

---

## 5. Dashboard 包含哪些页面

当前六页都已经接通：

- `Overview`
- `Data Browser`
- `Ingestion Manager`
- `Ingestion Traces`
- `Query Traces`
- `Evaluation Panel`

其中：

- `Data Browser` 可以查看文档摘要并删除文档
- `Ingestion Manager` 可以上传 PDF 并直接走摄取链路
- `Ingestion Traces / Query Traces` 会读取 `logs/traces.jsonl`
- `Evaluation Panel` 会读取 `data/evaluation/*.json`

---

## 6. MCP 工具列表

当前 MCP Server 暴露了 3 个最小工具：

- `query_knowledge_hub`
- `list_collections`
- `get_document_summary`

它们分别对应：

- 查询知识库
- 列出集合
- 返回文档摘要

---

## 7. 测试怎么跑

### 7.1 单元测试

```powershell
..\.venv\Scripts\python.exe -m pytest tests\unit -q
```

---

### 7.2 集成测试

```powershell
..\.venv\Scripts\python.exe -m pytest tests\integration -q
```

---

### 7.3 E2E 冒烟测试

```powershell
..\.venv\Scripts\python.exe -m pytest tests\e2e -q
```

---

### 7.4 全量测试

```powershell
..\.venv\Scripts\python.exe -m pytest tests -q
```

当前最近一次全量结果：

- `44 passed`

---

## 8. 关键文档

如果你想快速理解这个复现版，建议按这个顺序看：

- `阶段总览图.md`
- `复现规划.md`
- `复现任务清单.md`
- `功能矩阵.md`
- `验收矩阵.md`
- `Phase8测试顺序图.md`

---

## 9. 当前已知边界

这份复现版已经可用，但还保留了几个明确边界：

- 还没有做真正的 rerank
- 当前只实现了本地 `simple rerank`，还没有 cross-encoder / LLM rerank
- `ChunkRefiner` 当前只实现了规则版清洗，还没有 LLM 增强版
- `MetadataEnricher` 当前只实现了规则版增强，还没有 LLM 增强版
- 已经做了图片落盘和本地索引，但还没有 `image_captioner`
- 还没有接生成式回答
- 还没有做多模态返回
- 还没有拆成自己独立的 `.venv`

也就是说，这份复现版更偏“最小可用的工程闭环”，不是完整生产版。

---

## 10. 当前状态一句话总结

你现在可以把这份 `repro/` 理解成：

> 一套已经完成了“摄取、检索、MCP、Dashboard、Trace、评估、文档管理、测试收口”的最小 Modular RAG 复现版。
