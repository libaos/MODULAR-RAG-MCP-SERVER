"""复现版项目的配置加载模块。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


def get_repro_root() -> Path:
    """返回复现版项目根目录。"""
    return Path(__file__).resolve().parents[2]


def resolve_path(path_value: str) -> Path:
    """将配置里的相对路径解析为基于复现根目录的绝对路径。"""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return get_repro_root() / path


@dataclass(slots=True)
class LLMSettings:
    """文本模型配置。"""
    provider: str = "ollama"
    model: str = "qwen2.5:3b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 2048


@dataclass(slots=True)
class EmbeddingSettings:
    """Embedding 模型配置。"""
    provider: str = "ollama"
    model: str = "nomic-embed-text"
    dimensions: int = 768
    base_url: str = "http://localhost:11434"


@dataclass(slots=True)
class VisionLLMSettings:
    """视觉模型配置。"""
    enabled: bool = False
    provider: str = "openai"
    model: str = "gpt-4o"
    max_image_size: int = 2048


@dataclass(slots=True)
class VectorStoreSettings:
    """向量库存储配置。"""
    provider: str = "chroma"
    persist_directory: str = "./data/db/chroma"
    collection_name: str = "default"


@dataclass(slots=True)
class RetrievalSettings:
    """检索参数配置。"""
    dense_top_k: int = 20
    sparse_top_k: int = 20
    fusion_top_k: int = 10
    rrf_k: int = 60


@dataclass(slots=True)
class RerankSettings:
    """Rerank 参数配置。"""
    enabled: bool = False
    provider: str = "none"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5


@dataclass(slots=True)
class EvaluationSettings:
    """评估配置。"""
    enabled: bool = False
    provider: str = "custom"
    metrics: List[str] = field(default_factory=lambda: ["hit_rate", "mrr", "faithfulness"])


@dataclass(slots=True)
class ObservabilitySettings:
    """日志和 Trace 配置。"""
    log_level: str = "INFO"
    trace_enabled: bool = True
    trace_file: str = "./logs/traces.jsonl"
    structured_logging: bool = True


@dataclass(slots=True)
class RefinerSettings:
    """通用的开关型增强配置。"""
    use_llm: bool = False


@dataclass(slots=True)
class IngestionSettings:
    """摄取链路配置。"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    splitter: str = "recursive"
    batch_size: int = 100
    chunk_refiner: RefinerSettings = field(default_factory=RefinerSettings)
    metadata_enricher: RefinerSettings = field(default_factory=RefinerSettings)


@dataclass(slots=True)
class Settings:
    """应用总配置对象。"""
    llm: LLMSettings = field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    vision_llm: VisionLLMSettings = field(default_factory=VisionLLMSettings)
    vector_store: VectorStoreSettings = field(default_factory=VectorStoreSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    rerank: RerankSettings = field(default_factory=RerankSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    observability: ObservabilitySettings = field(default_factory=ObservabilitySettings)
    ingestion: IngestionSettings = field(default_factory=IngestionSettings)


def _read_yaml(path: Path) -> Dict[str, Any]:
    """读取 YAML 文件并确保根节点是字典。"""
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at root of config file: {path}")
    return data


def load_settings(config_path: str | None = None) -> Settings:
    """从 YAML 加载配置并映射到 dataclass。

    当前阶段先保持实现直接、可读，后面再补更严格的校验。
    """
    path = Path(config_path) if config_path else get_repro_root() / "config" / "settings.yaml"
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    raw = _read_yaml(path)

    # 摄取模块里有两层子配置，这里先手动拆出来，避免过早引入复杂配置框架。
    ingestion_raw = raw.get("ingestion", {})
    chunk_refiner_raw = ingestion_raw.get("chunk_refiner", {})
    metadata_enricher_raw = ingestion_raw.get("metadata_enricher", {})

    return Settings(
        llm=LLMSettings(**raw.get("llm", {})),
        embedding=EmbeddingSettings(**raw.get("embedding", {})),
        vision_llm=VisionLLMSettings(**raw.get("vision_llm", {})),
        vector_store=VectorStoreSettings(**raw.get("vector_store", {})),
        retrieval=RetrievalSettings(**raw.get("retrieval", {})),
        rerank=RerankSettings(**raw.get("rerank", {})),
        evaluation=EvaluationSettings(**raw.get("evaluation", {})),
        observability=ObservabilitySettings(**raw.get("observability", {})),
        ingestion=IngestionSettings(
            chunk_size=ingestion_raw.get("chunk_size", 1000),
            chunk_overlap=ingestion_raw.get("chunk_overlap", 200),
            splitter=ingestion_raw.get("splitter", "recursive"),
            batch_size=ingestion_raw.get("batch_size", 100),
            chunk_refiner=RefinerSettings(**chunk_refiner_raw),
            metadata_enricher=RefinerSettings(**metadata_enricher_raw),
        ),
    )
