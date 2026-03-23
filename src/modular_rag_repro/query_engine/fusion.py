"""RRF 融合器。

当前阶段实现最小可用的 Reciprocal Rank Fusion：

- 输入多路 `RetrievalResult` 排名列表
- 按 rank 位置累加 RRF 分数
- 输出统一的融合结果

这里故意不做分数归一化，因为 RRF 本来就是“只看排名，不看原始分值”的方法。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from modular_rag_repro.types import RetrievalResult


class RRFFusion:
    """最小可用的 RRF 融合器。"""

    def __init__(self, k: int = 60) -> None:
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k 必须是正整数，收到 {k}")
        self.k = k

    def fuse(
        self,
        ranking_lists: Iterable[List[RetrievalResult]],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """融合多路检索结果。"""
        lists = [items for items in ranking_lists if items]
        if not lists:
            return []

        scores: Dict[str, float] = {}
        first_seen: Dict[str, RetrievalResult] = {}

        for ranking_list in lists:
            for rank, item in enumerate(ranking_list, start=1):
                scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + self.rrf_score(rank)
                if item.chunk_id not in first_seen:
                    first_seen[item.chunk_id] = item

        fused: List[RetrievalResult] = []
        for chunk_id, score in scores.items():
            original = first_seen[chunk_id]
            fused.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=score,
                    text=original.text,
                    metadata=dict(original.metadata),
                )
            )

        fused.sort(key=lambda item: (-item.score, item.chunk_id))
        if top_k is not None and top_k > 0:
            fused = fused[:top_k]
        return fused

    def rrf_score(self, rank: int) -> float:
        """计算单个 rank 的 RRF 贡献分。"""
        if rank <= 0:
            raise ValueError(f"rank 必须大于 0，收到 {rank}")
        return 1.0 / (self.k + rank)
