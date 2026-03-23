"""BM25 稀疏索引器。

当前阶段只解决一件事：

- 把 `Chunk` 列表构建成可查询的 BM25 索引

它现在负责：

- 文本分词
- 统计词频 / 文档频率
- 计算 BM25 分数
- 把索引落到磁盘
- 根据关键词查询返回 `RetrievalResult`

它现在还不负责：

- 增量更新优化
- 删除文档后的重建
- 与 Dense 检索融合
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import jieba

from modular_rag_repro.settings import resolve_path
from modular_rag_repro.types import Chunk, RetrievalResult

# 避免 jieba 在首次分词时把初始化信息刷到终端输出里。
jieba.setLogLevel(logging.ERROR)


class BM25Indexer:
    """最小可用的 BM25 索引器。"""

    def __init__(self, index_dir: str = "data/db/bm25", k1: float = 1.5, b: float = 0.75) -> None:
        self.index_dir = resolve_path(index_dir)
        self.k1 = k1
        self.b = b
        self._index: Dict[str, Dict[str, Any]] = {}
        self._documents: Dict[str, Dict[str, Any]] = {}
        self._metadata: Dict[str, Any] = {}

        if self.k1 <= 0:
            raise ValueError("k1 必须大于 0")
        if not 0 <= self.b <= 1:
            raise ValueError("b 必须位于 0 到 1 之间")

    def build(self, chunks: List[Chunk], collection: str = "default") -> None:
        """基于 chunk 列表构建 BM25 索引。"""
        if not chunks:
            raise ValueError("不能基于空 chunk 列表构建 BM25 索引")

        doc_entries: List[Dict[str, Any]] = []
        doc_freq: Dict[str, int] = {}

        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            if not tokens:
                continue

            term_freq = Counter(tokens)
            doc_length = len(tokens)

            doc_entries.append(
                {
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "term_freq": dict(term_freq),
                    "doc_length": doc_length,
                }
            )

            for term in term_freq.keys():
                doc_freq[term] = doc_freq.get(term, 0) + 1

        if not doc_entries:
            raise ValueError("所有 chunk 在分词后都为空，无法构建 BM25 索引")

        num_docs = len(doc_entries)
        avg_doc_length = sum(item["doc_length"] for item in doc_entries) / num_docs

        index: Dict[str, Dict[str, Any]] = {}
        documents: Dict[str, Dict[str, Any]] = {}

        for item in doc_entries:
            documents[item["chunk_id"]] = {
                "text": item["text"],
                "metadata": item["metadata"],
                "doc_length": item["doc_length"],
            }

        for term, df in doc_freq.items():
            postings: List[Dict[str, Any]] = []
            for item in doc_entries:
                tf = item["term_freq"].get(term, 0)
                if tf > 0:
                    postings.append(
                        {
                            "chunk_id": item["chunk_id"],
                            "tf": tf,
                            "doc_length": item["doc_length"],
                        }
                    )

            index[term] = {
                "idf": self._calculate_idf(num_docs, df),
                "df": df,
                "postings": postings,
            }

        self._index = index
        self._documents = documents
        self._metadata = {
            "collection": collection,
            "num_docs": num_docs,
            "avg_doc_length": avg_doc_length,
            "total_terms": len(index),
            "k1": self.k1,
            "b": self.b,
        }

        self._save(collection)

    def load(self, collection: str = "default") -> bool:
        """从磁盘加载索引。"""
        path = self._get_index_path(collection)
        if not path.exists():
            return False

        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        self._metadata = payload["metadata"]
        self._documents = payload["documents"]
        self._index = payload["index"]
        return True

    def query(self, query_text: str, top_k: int = 10) -> List[RetrievalResult]:
        """执行 BM25 关键词检索。"""
        if not self._index:
            raise ValueError("索引尚未加载，请先调用 build() 或 load()")
        if not query_text or not query_text.strip():
            raise ValueError("query_text 不能为空")
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        query_terms = self._tokenize(query_text)
        if not query_terms:
            return []

        scores: Dict[str, float] = {}
        avg_doc_length = float(self._metadata["avg_doc_length"])

        for term in query_terms:
            term_info = self._index.get(term)
            if not term_info:
                continue

            idf = float(term_info["idf"])
            for posting in term_info["postings"]:
                chunk_id = posting["chunk_id"]
                tf = int(posting["tf"])
                doc_length = int(posting["doc_length"])
                score = self._calculate_bm25_score(tf, doc_length, avg_doc_length, idf)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + score

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

        results: List[RetrievalResult] = []
        for chunk_id, score in ranked:
            doc_info = self._documents[chunk_id]
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=score,
                    text=doc_info["text"],
                    metadata=doc_info["metadata"],
                )
            )
        return results

    def _save(self, collection: str) -> None:
        """把索引保存到磁盘。"""
        path = self._get_index_path(collection)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": self._metadata,
            "documents": self._documents,
            "index": self._index,
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    def _get_index_path(self, collection: str) -> Path:
        """返回某个 collection 的索引文件路径。"""
        return self.index_dir / collection / "bm25_index.json"

    def _tokenize(self, text: str) -> List[str]:
        """中英混合的最小分词策略。

        规则：

        - 英文 / 数字走正则切词
        - 中文走 jieba
        - 最终统一转小写
        """
        text = text.lower().strip()
        if not text:
            return []

        english_tokens = re.findall(r"[a-z0-9_]+", text)
        chinese_like_tokens = [
            token.strip().lower()
            for token in jieba.lcut(text)
            if token.strip() and re.search(r"[\u4e00-\u9fff]", token)
        ]

        return english_tokens + chinese_like_tokens

    def _calculate_idf(self, num_docs: int, doc_freq: int) -> float:
        """计算 BM25 的 IDF。"""
        return math.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

    def _calculate_bm25_score(
        self,
        tf: int,
        doc_length: int,
        avg_doc_length: float,
        idf: float,
    ) -> float:
        """计算单个 term 对单个文档的 BM25 贡献分。"""
        denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / max(avg_doc_length, 1e-9)))
        return idf * (tf * (self.k1 + 1)) / denominator
