"""查询预处理器。

当前阶段先做最小能力：

- 清理原始 query
- 提取关键词
- 解析最简单的 `key:value` 过滤条件

这样做的目的不是“让模型更聪明”，而是先把查询链路里的输入标准化，
给后面的 Dense / Sparse 检索统一入口。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import jieba

from modular_rag_repro.types import ProcessedQuery

# 保持和 BM25 一样的静默分词体验，避免首次调用时刷终端日志。
jieba.setLogLevel(20)


class QueryProcessor:
    """最小可用的查询预处理器。"""

    _EN_STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "be",
        "for",
        "how",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "this",
        "to",
        "what",
        "with",
    }

    _ZH_STOPWORDS = {
        "一下",
        "什么",
        "这个",
        "以及",
        "你",
        "我",
        "它",
        "了",
        "呢",
        "吗",
        "和",
        "在",
        "是",
        "有",
        "的",
        "讲",
        "请问",
        "一下",
        "文档",
    }

    _FILTER_PATTERN = re.compile(r"(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(?P<value>[^\s]+)")

    def process(self, query_text: str) -> ProcessedQuery:
        """把用户输入转成结构化查询对象。"""
        original = query_text.strip()
        if not original:
            raise ValueError("query 不能为空")

        filters = self._parse_filters(original)
        normalized_text = self._remove_filter_tokens(original).strip()
        if not normalized_text:
            normalized_text = original

        keywords = self._extract_keywords(normalized_text)
        return ProcessedQuery(
            original_text=original,
            normalized_text=normalized_text,
            keywords=keywords,
            filters=filters,
        )

    def _parse_filters(self, text: str) -> Dict[str, Any]:
        """解析最简单的 `key:value` 过滤语法。"""
        filters: Dict[str, Any] = {}
        for match in self._FILTER_PATTERN.finditer(text):
            key = match.group("key")
            value = match.group("value")
            filters[key] = self._coerce_filter_value(value)
        return filters

    def _remove_filter_tokens(self, text: str) -> str:
        """把 `key:value` 片段从原 query 中去掉。"""
        return self._FILTER_PATTERN.sub(" ", text)

    def _extract_keywords(self, text: str) -> List[str]:
        """提取中英文混合关键词。"""
        lowered = text.lower().strip()
        if not lowered:
            return []

        english_tokens = re.findall(r"[a-z0-9_]+", lowered)
        chinese_tokens = [
            token.strip().lower()
            for token in jieba.lcut(lowered)
            if token.strip() and re.search(r"[\u4e00-\u9fff]", token)
        ]

        keywords: List[str] = []
        seen = set()
        for token in english_tokens + chinese_tokens:
            if token in self._EN_STOPWORDS or token in self._ZH_STOPWORDS:
                continue
            if len(token) == 1 and not re.search(r"[\u4e00-\u9fff]", token):
                continue
            if token in seen:
                continue
            seen.add(token)
            keywords.append(token)

        return keywords

    def _coerce_filter_value(self, value: str) -> Any:
        """把过滤值做最小类型转换。"""
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False

        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value
