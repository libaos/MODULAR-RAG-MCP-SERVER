"""PDF 读取器。

当前阶段只解决一件事：

- 把 PDF 文件读成一个 `Document`

实现策略尽量简单直接：

- 使用 `PyMuPDF(fitz)` 提取文本
- 可选提取图片元数据
- 生成稳定的 `doc_id` 和 `doc_hash`

后面如果要追求和参考实现更接近，再补 Markdown 转换、图片占位符、
更复杂的元数据抽取都可以。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List

import fitz

from modular_rag_repro.types import Document


class PdfLoader:
    """最小可用的 PDF 读取器。

    参数:
        extract_images: 是否尝试记录图片信息。
        image_storage_dir: 预留给后续图片落盘使用，当前阶段先只记录路径。
    """

    def __init__(
        self,
        extract_images: bool = True,
        image_storage_dir: str | Path = "data/images",
    ) -> None:
        self.extract_images = extract_images
        self.image_storage_dir = Path(image_storage_dir)

    def load(self, file_path: str | Path) -> Document:
        """读取 PDF 并转换为 `Document`。

        返回值中的 `text` 是整份文档拼接后的纯文本，
        后续会由 Chunker 再切成多个块。
        """
        path = Path(file_path).resolve()
        self._validate_pdf(path)

        doc_hash = self._compute_sha256(path)
        doc_id = f"doc_{doc_hash[:16]}"

        pdf = fitz.open(path)
        try:
            page_texts = [self._extract_page_text(page) for page in pdf]
            text = "\n\n".join(part for part in page_texts if part.strip()).strip()

            metadata: Dict[str, Any] = {
                "source_path": str(path),
                "doc_type": "pdf",
                "doc_hash": doc_hash,
                "page_count": len(pdf),
                "title": self._extract_title(page_texts),
            }

            if self.extract_images:
                metadata["images"] = self._extract_image_metadata(pdf, doc_hash)

            return Document(
                id=doc_id,
                text=text,
                source_path=str(path),
                metadata=metadata,
            )
        finally:
            pdf.close()

    def _validate_pdf(self, path: Path) -> None:
        """校验输入路径确实存在且是 PDF。"""
        if not path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {path}")
        if not path.is_file():
            raise ValueError(f"输入路径不是文件: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"文件不是 PDF: {path}")

    def _extract_page_text(self, page: fitz.Page) -> str:
        """提取单页文本。

        当前阶段直接用 `get_text()`，先保证能读出来。
        """
        return page.get_text().strip()

    def _extract_image_metadata(self, pdf: fitz.Document, doc_hash: str) -> List[Dict[str, Any]]:
        """提取图片元数据。

        当前先不真正落盘图片文件，只记录：

        - image_id
        - page
        - 图片序号
        - 预期存储目录
        """
        images: List[Dict[str, Any]] = []
        image_dir = self.image_storage_dir / doc_hash

        for page_index, page in enumerate(pdf):
            image_list = page.get_images(full=True)
            for image_index, image_info in enumerate(image_list, start=1):
                xref = image_info[0]
                images.append(
                    {
                        "id": self._build_image_id(doc_hash, page_index + 1, image_index),
                        "xref": xref,
                        "page": page_index + 1,
                        "index": image_index,
                        "storage_dir": str(image_dir),
                    }
                )

        return images

    def _extract_title(self, page_texts: List[str]) -> str:
        """从前几页文本里猜一个标题。

        规则很简单：

        - 找到第一条非空行
        - 去掉首尾空白
        - 最长只保留 120 个字符
        """
        for page_text in page_texts[:3]:
            for line in page_text.splitlines():
                candidate = line.strip()
                if candidate:
                    return candidate[:120]
        return "Untitled PDF"

    def _compute_sha256(self, path: Path) -> str:
        """计算文件 SHA256，用于幂等与文档唯一标识。"""
        sha256 = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _build_image_id(self, doc_hash: str, page: int, index: int) -> str:
        """生成稳定的图片 ID。"""
        return f"{doc_hash[:8]}_{page}_{index}"
