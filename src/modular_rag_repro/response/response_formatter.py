"""查询响应格式化器。"""

from __future__ import annotations

from typing import List

from modular_rag_repro.response.multimodal_assembler import MultimodalAssembler
from modular_rag_repro.settings import Settings
from modular_rag_repro.types import QueryResponse, QueryResponseItem, RetrievalResult


class ResponseFormatter:
    """最小可用的查询响应格式化器。"""

    def __init__(self, settings: Settings, preview_limit: int = 180) -> None:
        self.settings = settings
        self.preview_limit = preview_limit
        self.multimodal = MultimodalAssembler(settings)

    def format(
        self,
        query: str,
        collection: str,
        results: List[RetrievalResult],
    ) -> QueryResponse:
        """把检索结果格式化为稳定结构。"""
        items: List[QueryResponseItem] = []
        for rank, result in enumerate(results, start=1):
            images = self.multimodal.extract_images(result)
            items.append(
                QueryResponseItem(
                    rank=rank,
                    chunk_id=result.chunk_id,
                    score=result.score,
                    source_path=str(result.metadata.get("source_path", "(unknown)")),
                    chunk_index=result.metadata.get("chunk_index", "(unknown)"),
                    preview=self._build_preview(result.text),
                    image_count=len(images),
                    images=images,
                )
            )

        summary = self._build_summary(query=query, collection=collection, items=items)
        metadata = {
            "query": query,
            "collection": collection,
            "result_count": len(items),
            "top_chunk_id": items[0].chunk_id if items else None,
            "total_image_count": sum(item.image_count for item in items),
        }
        return QueryResponse(
            query=query,
            collection=collection,
            result_count=len(items),
            summary=summary,
            items=items,
            metadata=metadata,
        )

    def render_text(self, response: QueryResponse) -> str:
        """把结构化响应渲染成终端可读文本。"""
        lines = []
        lines.append("=" * 60)
        lines.append("FORMATTED RESPONSE")
        lines.append("=" * 60)
        lines.append(f"summary={response.summary}")

        if not response.items:
            lines.append("[INFO] 没有可展示的命中条目")
            lines.append("=" * 60)
            return "\n".join(lines)

        for item in response.items:
            lines.append(f"[{item.rank}] score={item.score:.4f} id={item.chunk_id}")
            lines.append(f"    source={item.source_path}")
            lines.append(f"    chunk_index={item.chunk_index}")
            lines.append(f"    image_count={item.image_count}")
            if item.images:
                caption = item.images[0].get("caption")
                if caption:
                    lines.append(f"    first_image_caption={caption}")
            lines.append(f"    preview={item.preview}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def _build_summary(self, query: str, collection: str, items: List[QueryResponseItem]) -> str:
        """构建最终结果摘要。"""
        if not items:
            return f"在集合 {collection} 中没有找到与“{query}”相关的结果。"

        top_item = items[0]
        return (
            f"在集合 {collection} 中找到 {len(items)} 条结果，"
            f"首条命中来自 {top_item.source_path} 的 chunk {top_item.chunk_index}"
            f"（关联图片 {top_item.image_count} 张）。"
        )

    def _build_preview(self, text: str) -> str:
        """把正文压缩成短预览。"""
        compact = " ".join(text.split())
        if len(compact) <= self.preview_limit:
            return compact
        return compact[: self.preview_limit - 3] + "..."
