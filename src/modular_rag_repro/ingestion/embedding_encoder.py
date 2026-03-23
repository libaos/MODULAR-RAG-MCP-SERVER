"""Dense Embedding 编码器。

当前阶段只解决一件事：

- 把 `Chunk` 列表编码成向量列表

实现策略：

- 先支持 `ollama`
- 优先调用 `/api/embed` 批量接口
- 如果本机 Ollama 版本较旧，再回退到 `/api/embeddings` 单条接口

这一步只负责“生成向量”，还不负责：

- 把向量写入 Chroma
- 和 BM25 融合
- 建立检索索引
"""

from __future__ import annotations

from typing import List

import requests

from modular_rag_repro.settings import Settings
from modular_rag_repro.types import Chunk


class EmbeddingEncoder:
    """最小可用的向量编码器。"""

    def __init__(self, settings: Settings, batch_size: int | None = None, timeout: float = 60.0) -> None:
        self.settings = settings
        self.provider = settings.embedding.provider.lower()
        self.model = settings.embedding.model
        self.base_url = settings.embedding.base_url.rstrip("/")
        self.batch_size = batch_size or settings.ingestion.batch_size
        self.timeout = timeout
        self.session = requests.Session()

        if self.batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")

    def encode_chunks(self, chunks: List[Chunk]) -> List[List[float]]:
        """把 chunk 列表编码成向量列表。"""
        if not chunks:
            raise ValueError("不能编码空 chunk 列表")
        texts = [chunk.text for chunk in chunks]
        return self.encode_texts(texts)

    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """把文本列表编码成向量列表。"""
        if not texts:
            raise ValueError("不能编码空文本列表")

        for index, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"第 {index} 条文本为空，无法生成向量")

        if self.provider != "ollama":
            raise ValueError(f"当前阶段只支持 ollama embedding，收到 provider={self.provider}")

        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._encode_batch_with_ollama(batch))

        if len(vectors) != len(texts):
            raise RuntimeError(f"向量数量不匹配: texts={len(texts)} vectors={len(vectors)}")

        self._validate_dimensions(vectors)
        return vectors

    def _encode_batch_with_ollama(self, batch_texts: List[str]) -> List[List[float]]:
        """优先走 Ollama 批量 embedding 接口。"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": batch_texts},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings")
            if not isinstance(embeddings, list):
                raise RuntimeError("Ollama /api/embed 响应中缺少 embeddings 字段")
            return embeddings
        except Exception:
            # 某些 Ollama 版本没有批量接口，回退到单条 embedding。
            return [self._encode_single_with_legacy_endpoint(text) for text in batch_texts]

    def _encode_single_with_legacy_endpoint(self, text: str) -> List[float]:
        """兼容旧版 Ollama 的单条 embedding 接口。"""
        response = self.session.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError("Ollama /api/embeddings 响应中缺少 embedding 字段")
        return embedding

    def _validate_dimensions(self, vectors: List[List[float]]) -> None:
        """校验所有向量维度一致。"""
        if not vectors:
            return

        expected_dim = len(vectors[0])
        if expected_dim == 0:
            raise RuntimeError("返回的向量维度为 0")

        for index, vector in enumerate(vectors):
            if len(vector) != expected_dim:
                raise RuntimeError(
                    f"向量维度不一致: 第 0 条={expected_dim}, 第 {index} 条={len(vector)}"
                )
