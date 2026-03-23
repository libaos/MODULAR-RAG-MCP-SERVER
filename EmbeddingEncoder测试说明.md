# EmbeddingEncoder 测试说明

## 这个测试是测什么

当前要测试的是这条链路：

```text
PDF -> PdfLoader -> Chunker -> EmbeddingEncoder
```

也就是验证：

- PDF 能读出来
- 文档能切成 chunk
- chunk 能生成向量

## 进入目录

```powershell
cd "D:\OneDrive - stu.scau.edu.cn\桌面\workspace\MODULAR-RAG-MCP-SERVER\repro"
```

## 测试 1：测完整最小链路

```powershell
..\.venv\Scripts\python.exe -c "from pathlib import Path; import sys; sys.path.insert(0, str(Path('src').resolve())); from modular_rag_repro.settings import load_settings; from modular_rag_repro.ingestion import PdfLoader, DocumentChunker, EmbeddingEncoder; settings = load_settings(); settings.ingestion.chunk_size = 120; settings.ingestion.chunk_overlap = 20; doc = PdfLoader(extract_images=False).load(Path('..') / 'tests' / 'fixtures' / 'sample_documents' / 'simple.pdf'); chunks = DocumentChunker(settings).split_document(doc); vectors = EmbeddingEncoder(settings, batch_size=2).encode_chunks(chunks); print('chunk_count=', len(chunks)); print('vector_count=', len(vectors)); print('vector_dim=', len(vectors[0])); print('first_chunk_id=', chunks[0].id); print('first_vector_head=', [round(x, 4) for x in vectors[0][:5]])"
```

## 预期输出

输出大致会像这样：

```text
chunk_count= 4
vector_count= 4
vector_dim= 768
first_chunk_id= doc_xxx_0000_xxxx
first_vector_head= [0.0205, 0.0589, -0.1982, 0.0643, 0.0054]
```

## 每个输出字段是什么意思

- `chunk_count`
  - 文档被切成了多少个 chunk

- `vector_count`
  - 实际生成了多少个向量

- `vector_dim`
  - 每个向量是多少维

- `first_chunk_id`
  - 第一个 chunk 的 ID

- `first_vector_head`
  - 第一个向量前几个值，用来确认返回的不是空结果

## 如何判断通过

只要满足下面三条，就说明这一步通过：

1. `vector_count == chunk_count`
2. `vector_dim == 768`
3. `first_vector_head` 里确实有一串浮点数

## 测试 2：只测编码器本身

如果你不想连 PDF 和 chunk 一起测，只想确认编码器能不能调通 Ollama，可以跑这个：

```powershell
..\.venv\Scripts\python.exe -c "from pathlib import Path; import sys; sys.path.insert(0, str(Path('src').resolve())); from modular_rag_repro.settings import load_settings; from modular_rag_repro.ingestion import EmbeddingEncoder; settings = load_settings(); vectors = EmbeddingEncoder(settings, batch_size=1).encode_texts(['hello world']); print('vector_count=', len(vectors)); print('vector_dim=', len(vectors[0])); print('sample=', [round(x, 4) for x in vectors[0][:5]])"
```

## 预期输出

```text
vector_count= 1
vector_dim= 768
sample= [-0.0068, -0.0013, -0.1714, 0.0085, 0.0058]
```

## 常见问题排查

### 1. 没有向量返回

先确认 Ollama 正在运行：

```powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

### 2. 模型不存在

确认本地已经拉了 embedding 模型：

- `nomic-embed-text:latest`

### 3. 配置不对

确认配置文件：

- `repro/config/settings.yaml`

至少要保证：

```yaml
embedding:
  provider: "ollama"
  model: "nomic-embed-text"
```

## 一句话总结

这一步测通，就说明下面这条链路已经成立：

```text
PDF -> Document -> Chunks -> Embeddings
```
