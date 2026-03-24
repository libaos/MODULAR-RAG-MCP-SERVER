"""复现版项目的核心数据类型。"""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4


@dataclass(slots=True)
class Document:
    """原始文档对象。"""
    id: str
    text: str
    source_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    """切分后的文本块对象。"""
    id: str
    document_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalResult:
    """一次检索返回的结果项。"""
    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProcessedQuery:
    """查询预处理后的结果。"""

    original_text: str
    normalized_text: str
    keywords: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TraceStage:
    """Trace 中的单个阶段记录。"""
    name: str
    payload: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: Optional[float] = None


@dataclass(slots=True)
class TraceContext:
    """一次摄取或查询的 Trace 容器。"""
    trace_type: str
    trace_id: str = field(default_factory=lambda: f"trace_{uuid4().hex[:12]}")
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    stages: List[TraceStage] = field(default_factory=list)

    def record_stage(
        self,
        name: str,
        payload: Optional[Dict[str, Any]] = None,
        elapsed_ms: Optional[float] = None,
    ) -> None:
        """追加一条阶段记录。"""
        self.stages.append(
            TraceStage(
                name=name,
                payload=payload or {},
                elapsed_ms=elapsed_ms,
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化字典。"""
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "stages": [asdict(stage) for stage in self.stages],
        }


@dataclass(slots=True)
class EvaluationQueryResult:
    """单条评估样本的结果。"""
    query: str
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    elapsed_ms: float = 0.0


@dataclass(slots=True)
class EvaluationReport:
    """评估报告对象。"""
    evaluator_name: str
    test_set_path: str
    query_results: List[EvaluationQueryResult] = field(default_factory=list)
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    total_elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化字典。"""
        return {
            "evaluator_name": self.evaluator_name,
            "test_set_path": self.test_set_path,
            "query_results": [asdict(item) for item in self.query_results],
            "aggregate_metrics": self.aggregate_metrics,
            "total_elapsed_ms": self.total_elapsed_ms,
        }


@dataclass(slots=True)
class QueryResponseItem:
    """格式化后的单条查询结果。"""

    rank: int
    chunk_id: str
    score: float
    source_path: str
    chunk_index: Any
    preview: str
    image_count: int = 0
    images: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class QueryResponse:
    """格式化后的查询响应。"""

    query: str
    collection: str
    result_count: int
    summary: str
    items: List[QueryResponseItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化字典。"""
        return {
            "query": self.query,
            "collection": self.collection,
            "result_count": self.result_count,
            "summary": self.summary,
            "items": [asdict(item) for item in self.items],
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class DeleteDocumentResult:
    """文档删除结果。"""

    doc_id: str
    collection: str
    deleted_chunk_ids: List[str] = field(default_factory=list)
    chroma_deleted_count: int = 0
    bm25_remaining_count: int = 0
    collection_removed: bool = False
    rebuilt: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化字典。"""
        return {
            "doc_id": self.doc_id,
            "collection": self.collection,
            "deleted_chunk_ids": self.deleted_chunk_ids,
            "chroma_deleted_count": self.chroma_deleted_count,
            "bm25_remaining_count": self.bm25_remaining_count,
            "collection_removed": self.collection_removed,
            "rebuilt": self.rebuilt,
        }
