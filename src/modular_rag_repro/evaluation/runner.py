"""Golden Set 评估执行器。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional

from modular_rag_repro.query_engine import QueryWorkflow
from modular_rag_repro.settings import Settings, resolve_path
from modular_rag_repro.types import EvaluationQueryResult, EvaluationReport


@dataclass(slots=True)
class GoldenTestCase:
    """单条 Golden Set 样本。"""

    query: str
    expected_chunk_ids: List[str]
    expected_sources: List[str]
    reference_answer: str = ""
    collection: Optional[str] = None


class EvaluationRunner:
    """最小可用的 Golden Set 评估器。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.query_workflow = QueryWorkflow(settings)
        self.output_dir = resolve_path("data/evaluation")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        test_set_path: str,
        collection: Optional[str] = None,
        top_k: int = 10,
        no_search: bool = False,
    ) -> EvaluationReport:
        """执行整套 Golden Set。"""
        path = self._resolve_test_set_path(test_set_path)
        cases = self._load_test_cases(path)

        started_at = perf_counter()
        query_results: List[EvaluationQueryResult] = []

        for case in cases:
            effective_collection = collection or case.collection or self.settings.vector_store.collection_name

            if no_search:
                query_results.append(
                    EvaluationQueryResult(
                        query=case.query,
                        retrieved_chunk_ids=[],
                        metrics={"hit_rate": 0.0, "mrr": 0.0},
                        elapsed_ms=0.0,
                    )
                )
                continue

            case_started = perf_counter()
            workflow_result = self.query_workflow.run(
                query=case.query,
                collection=effective_collection,
                top_k=top_k,
                no_rerank=True,
                source="evaluation",
            )
            elapsed_ms = (perf_counter() - case_started) * 1000.0
            retrieved_chunk_ids = [item.chunk_id for item in workflow_result.final_results]
            retrieved_sources = [str(item.metadata.get("source_path", "")) for item in workflow_result.final_results]

            metrics = self._calculate_metrics(
                expected_chunk_ids=case.expected_chunk_ids,
                expected_sources=case.expected_sources,
                retrieved_chunk_ids=retrieved_chunk_ids,
                retrieved_sources=retrieved_sources,
            )
            query_results.append(
                EvaluationQueryResult(
                    query=case.query,
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    metrics=metrics,
                    elapsed_ms=elapsed_ms,
                )
            )

        total_elapsed_ms = (perf_counter() - started_at) * 1000.0
        aggregate_metrics = self._aggregate_metrics(query_results)
        report = EvaluationReport(
            evaluator_name="golden_set_runner",
            test_set_path=str(path),
            query_results=query_results,
            aggregate_metrics=aggregate_metrics,
            total_elapsed_ms=total_elapsed_ms,
        )
        self._write_report(report)
        return report

    def _resolve_test_set_path(self, test_set_path: str) -> Path:
        """解析测试集路径。"""
        path = Path(test_set_path)
        if path.is_absolute() and path.exists():
            return path

        candidates = [
            resolve_path(test_set_path),
            resolve_path(f"./{test_set_path}"),
            resolve_path("../tests/fixtures/golden_test_set.json") if test_set_path == "tests/fixtures/golden_test_set.json" else None,
        ]
        for candidate in candidates:
            if candidate and candidate.exists():
                return candidate
        raise FileNotFoundError(f"Golden test set not found: {test_set_path}")

    def _load_test_cases(self, path: Path) -> List[GoldenTestCase]:
        """加载 JSON 测试集。"""
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        cases = []
        for item in payload.get("test_cases", []):
            cases.append(
                GoldenTestCase(
                    query=item["query"],
                    expected_chunk_ids=list(item.get("expected_chunk_ids", [])),
                    expected_sources=list(item.get("expected_sources", [])),
                    reference_answer=str(item.get("reference_answer", "")),
                    collection=item.get("collection"),
                )
            )
        return cases

    def _calculate_metrics(
        self,
        expected_chunk_ids: List[str],
        expected_sources: List[str],
        retrieved_chunk_ids: List[str],
        retrieved_sources: List[str],
    ) -> Dict[str, float]:
        """计算最小 IR 指标。"""
        matches = []
        for index, chunk_id in enumerate(retrieved_chunk_ids, start=1):
            if expected_chunk_ids and chunk_id in expected_chunk_ids:
                matches.append(index)
                continue
            if expected_sources and index - 1 < len(retrieved_sources) and retrieved_sources[index - 1] in expected_sources:
                matches.append(index)

        if not expected_chunk_ids and not expected_sources:
            return {"hit_rate": 0.0, "mrr": 0.0}

        hit_rate = 1.0 if matches else 0.0
        mrr = 1.0 / matches[0] if matches else 0.0
        return {"hit_rate": hit_rate, "mrr": mrr}

    def _aggregate_metrics(self, query_results: List[EvaluationQueryResult]) -> Dict[str, float]:
        """对所有 query 结果做均值聚合。"""
        if not query_results:
            return {"hit_rate": 0.0, "mrr": 0.0}

        metrics = {"hit_rate": 0.0, "mrr": 0.0}
        for item in query_results:
            metrics["hit_rate"] += item.metrics.get("hit_rate", 0.0)
            metrics["mrr"] += item.metrics.get("mrr", 0.0)

        count = len(query_results)
        return {
            "hit_rate": metrics["hit_rate"] / count,
            "mrr": metrics["mrr"] / count,
        }

    def _write_report(self, report: EvaluationReport) -> Path:
        """把评估报告写到 data/evaluation。"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = self.output_dir / f"evaluation-report-{timestamp}.json"
        payload = report.to_dict()
        payload["generated_at"] = datetime.now().isoformat()
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        return path
