"""复现版项目的核心数据类型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


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
            "trace_type": self.trace_type,
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
