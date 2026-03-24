"""最小 Ollama 文本/视觉调用客户端。"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Iterable, List

import requests


class OllamaClient:
    """对 Ollama `/api/generate` 的轻量封装。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.session = requests.Session()

    def generate(self, prompt: str, image_paths: Iterable[str] | None = None) -> str:
        """调用 Ollama 生成文本。"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        encoded_images = self._encode_images(image_paths or [])
        if encoded_images:
            payload["images"] = encoded_images

        response = self.session.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        text = str(data.get("response") or "").strip()
        if not text:
            raise RuntimeError("Ollama /api/generate 响应中缺少 response 文本")
        return text

    def _encode_images(self, image_paths: Iterable[str]) -> List[str]:
        """把图片文件编码成 base64，供 Ollama 视觉模型使用。"""
        encoded: List[str] = []
        for raw_path in image_paths:
            path = Path(raw_path)
            if not path.exists() or not path.is_file():
                continue
            encoded.append(base64.b64encode(path.read_bytes()).decode("utf-8"))
        return encoded
