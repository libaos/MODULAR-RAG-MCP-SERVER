"""Dashboard 数据服务。

这个模块把 Dashboard 需要的本地数据读取逻辑集中起来：

- 概览统计
- collection / document 浏览
- 上传后调用摄取链路
- Trace 与评估文件读取
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from modular_rag_repro.ingestion import IngestionPipeline, PipelineResult
from modular_rag_repro.management import DocumentManager
from modular_rag_repro.settings import Settings, load_settings, resolve_path
from modular_rag_repro.trace import TraceCollector
from modular_rag_repro.types import TraceContext


class DashboardService:
    """Dashboard 读写服务。"""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or load_settings()
        self.document_manager = DocumentManager(self.settings)
        self.catalog = self.document_manager.catalog
        self.trace_file = resolve_path(self.settings.observability.trace_file)
        self.evaluation_dir = resolve_path("data/evaluation")

    def get_overview(self) -> Dict[str, Any]:
        """返回概览页统计。"""
        collections = self.catalog.list_collections(include_stats=True)
        total_collections = len(collections)
        total_chroma = sum(item.chroma_count or 0 for item in collections)
        total_bm25 = sum(item.bm25_count or 0 for item in collections)
        documents = []
        for collection in collections:
            documents.extend(self.catalog.list_documents(collection.name))
        traces = self.read_traces()

        return {
            "total_collections": total_collections,
            "total_documents": len(documents),
            "total_chroma_vectors": total_chroma,
            "total_bm25_docs": total_bm25,
            "trace_count": len(traces),
            "collections": collections,
        }

    def list_collections(self) -> List[Dict[str, Any]]:
        """返回 collection 列表。"""
        items = []
        for item in self.catalog.list_collections(include_stats=True):
            items.append(
                {
                    "name": item.name,
                    "chroma_count": item.chroma_count,
                    "bm25_count": item.bm25_count,
                    "metadata": item.metadata,
                }
            )
        return items

    def list_documents(self, collection: Optional[str] = None) -> List[Dict[str, Any]]:
        """返回文档列表。"""
        documents = []
        target_collections = [collection] if collection else [item["name"] for item in self.list_collections()]
        for collection_name in target_collections:
            if not collection_name:
                continue
            for item in self.catalog.list_documents(collection_name):
                documents.append(
                    {
                        "doc_id": item.doc_id,
                        "collection": item.collection,
                        "title": item.title,
                        "source_path": item.source_path,
                        "chunk_count": item.chunk_count,
                        "summary": item.summary,
                        "tags": item.tags,
                        "metadata": item.metadata,
                    }
                )
        return documents

    def ingest_uploaded_file(self, file_name: str, file_bytes: bytes, collection: str) -> Dict[str, Any]:
        """保存上传文件并调用摄取链路。"""
        suffix = Path(file_name).suffix or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="repro-upload-") as tmp:
            tmp.write(file_bytes)
            temp_path = Path(tmp.name)

        try:
            pipeline = IngestionPipeline(self.settings, collection=collection)
            trace = TraceContext(
                trace_type="ingestion",
                metadata={
                    "source": "dashboard",
                    "collection": collection,
                    "file_path": str(temp_path),
                    "uploaded_file_name": file_name,
                },
            )
            result = pipeline.run(str(temp_path), trace=trace)
            trace.metadata["success"] = result.success
            trace.metadata["file_hash"] = result.file_hash
            trace.metadata["skipped"] = result.skipped
            trace.metadata["skip_reason"] = result.skip_reason
            trace.metadata["document_id"] = result.document.id if result.document else None
            trace.metadata["chunk_count"] = result.chunk_count
            trace.metadata["vector_count"] = result.vector_count
            trace.metadata["upserted_count"] = result.upserted_count
            if result.error:
                trace.metadata["error"] = result.error
            if self.settings.observability.trace_enabled:
                TraceCollector(self.settings.observability.trace_file).collect(trace)
            return self._pipeline_result_to_dict(result)
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def read_traces(self) -> List[Dict[str, Any]]:
        """读取 trace JSONL。"""
        if not self.trace_file.exists():
            return []

        records: List[Dict[str, Any]] = []
        with self.trace_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def read_traces_by_type(self, trace_type: str) -> List[Dict[str, Any]]:
        """按 trace_type 过滤 trace。"""
        return [record for record in self.read_traces() if record.get("trace_type") == trace_type]

    def delete_document(self, doc_id: str, collection: str) -> Dict[str, Any]:
        """删除某个文档，并返回删除结果。"""
        result = self.document_manager.delete_document(doc_id=doc_id, collection=collection)
        return result.to_dict()

    def list_evaluation_reports(self) -> List[Dict[str, Any]]:
        """读取评估报告目录。"""
        if not self.evaluation_dir.exists():
            return []

        reports: List[Dict[str, Any]] = []
        for path in sorted(self.evaluation_dir.rglob("*.json")):
            try:
                with path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
            except Exception:
                continue
            reports.append(
                {
                    "path": str(path),
                    "evaluator_name": payload.get("evaluator_name"),
                    "aggregate_metrics": payload.get("aggregate_metrics", {}),
                    "total_elapsed_ms": payload.get("total_elapsed_ms"),
                }
            )
        return reports

    def _pipeline_result_to_dict(self, result: PipelineResult) -> Dict[str, Any]:
        """把 PipelineResult 转成适合前端展示的字典。"""
        return {
            "success": result.success,
            "file_path": result.file_path,
            "document_id": result.document.id if result.document else None,
            "file_hash": result.file_hash,
            "chunk_count": result.chunk_count,
            "vector_count": result.vector_count,
            "upserted_count": result.upserted_count,
            "skipped": result.skipped,
            "skip_reason": result.skip_reason,
            "error": result.error,
            "stages": result.stages,
        }
