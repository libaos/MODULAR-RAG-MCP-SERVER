# repro

这个目录是“复现版实现区”。

仓库根目录继续作为参考实现和对照基线，`repro/` 负责你自己的功能等价重做。

## 使用规则

- 根目录代码只用来阅读、运行、对照行为。
- 新的复现代码只写在 `repro/` 下面。
- 除非明确需要，不要把复现代码混进根目录的 `src/`。

## 复现目标

按阶段复现一个功能等价的 Modular RAG MCP Server：

1. 文档摄取
2. 查询检索
3. MCP 工具
4. Dashboard
5. Trace 与评估

## 当前进度

当前已经完成到 `Phase 2` 的最小离线摄取闭环：

- `PDF -> Document -> Chunks -> Embeddings -> BM25 -> Chroma`

当前也已经进入 `Phase 3` 的第一步：

- `query -> QueryProcessor -> DenseRetriever + SparseRetriever -> RRF Fusion -> ResponseFormatter`

当前已经进入 `Phase 4` 的最小 MCP 闭环：

- `MCP Server -> ProtocolHandler -> query_knowledge_hub / list_collections / get_document_summary`

还没有完成的重点：

- SHA256 去重
- Query 主链路
- MCP tools
- Dashboard 六页
- Trace 与评估

## 从这里开始

先看这几份文件：

- `repro/复现规划.md`
- `repro/复现任务清单.md`
- `repro/功能矩阵.md`
- `repro/验收矩阵.md`

## 当前运行方式

当前这份复现版还没有自己的独立虚拟环境，临时复用外层项目的 `.venv`。

在 `repro/` 目录下可直接这样验证 Phase 1 骨架：

```powershell
..\.venv\Scripts\python.exe scripts\ingest.py --help
..\.venv\Scripts\python.exe scripts\query.py --help
..\.venv\Scripts\python.exe scripts\evaluate.py --help
..\.venv\Scripts\python.exe scripts\start_dashboard.py --help
```

如果后面要继续开发，建议再单独给 `repro/` 建自己的 `.venv`。
