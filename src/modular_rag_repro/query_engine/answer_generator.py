"""生成式最终回答。"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from modular_rag_repro.llm import OllamaClient
from modular_rag_repro.settings import Settings
from modular_rag_repro.types import RetrievalResult


class AnswerGenerator:
    """负责把最终检索结果变成最终回答。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.enabled = settings.answer_generation.enabled
        self.provider = settings.answer_generation.provider.lower()
        self.max_context_chunks = settings.answer_generation.max_context_chunks
        self.client = OllamaClient(
            base_url=settings.llm.base_url,
            model=settings.answer_generation.model or settings.llm.model,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            timeout=settings.answer_generation.timeout_seconds,
        )

    def generate(self, query: str, results: List[RetrievalResult]) -> Tuple[Optional[str], str, Dict[str, object]]:
        """生成最终回答，并返回 mode/metadata。"""
        if not self.enabled:
            return None, "disabled", {"provider": "none"}
        if not results:
            return None, "no_context", {"provider": self.provider}

        provider = self.provider
        if provider == "local":
            return self._generate_local(query, results), "generated", {"provider": "local"}
        if provider == "ollama":
            try:
                return self._generate_with_ollama(query, results), "generated", {"provider": "ollama"}
            except Exception as exc:
                return (
                    self._generate_local(query, results),
                    "fallback_local",
                    {
                        "provider": "ollama",
                        "fallback_provider": "local",
                        "error": str(exc),
                    },
                )

        return (
            self._generate_local(query, results),
            "fallback_local",
            {
                "provider": provider,
                "fallback_provider": "local",
                "error": f"unsupported provider: {provider}",
            },
        )

    def _generate_local(self, query: str, results: List[RetrievalResult]) -> str:
        """本地模板版最终回答。"""
        top_results = results[: max(self.max_context_chunks, 1)]
        evidence = []
        for item in top_results:
            source_path = str(item.metadata.get("source_path", "(unknown)"))
            chunk_index = item.metadata.get("chunk_index", "(unknown)")
            preview = " ".join(item.text.split())[:140]
            evidence.append(f"{source_path}#chunk{chunk_index}: {preview}")

        joined = "；".join(evidence)
        primary_source = str(top_results[0].metadata.get("source_path", "(unknown)"))
        return (
            f"根据检索结果，关于“{query}”的回答主要来自 {primary_source}。"
            f"综合前 {len(top_results)} 条命中，可以判断：{joined}"
        )

    def _generate_with_ollama(self, query: str, results: List[RetrievalResult]) -> str:
        """用 Ollama 生成最终回答。"""
        top_results = results[: max(self.max_context_chunks, 1)]
        context_lines = []
        for index, item in enumerate(top_results, start=1):
            context_lines.append(
                f"[{index}] source={item.metadata.get('source_path', '(unknown)')} "
                f"chunk={item.metadata.get('chunk_index', '(unknown)')}\n{item.text}"
            )
        prompt = (
            "你是一个严格基于检索结果回答问题的助手。\n"
            "只允许使用给定上下文，不要编造。\n\n"
            f"问题：{query}\n\n"
            "上下文：\n"
            f"{chr(10).join(context_lines)}\n\n"
            "请给出简洁中文回答，并点出主要来源。"
        )
        return self.client.generate(prompt)
