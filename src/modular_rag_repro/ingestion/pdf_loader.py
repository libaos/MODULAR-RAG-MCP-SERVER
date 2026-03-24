"""PDF 读取器。"""

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
            page_texts: List[str] = []
            images: List[Dict[str, Any]] = []

            for page_index, page in enumerate(pdf, start=1):
                page_images = self._extract_page_image_metadata(page, doc_hash, page_index) if self.extract_images else []
                images.extend(page_images)
                page_texts.append(self._build_page_text(page, page_images))

            text = "\n\n".join(part for part in page_texts if part.strip()).strip()

            metadata: Dict[str, Any] = {
                "source_path": str(path),
                "doc_type": "pdf",
                "doc_hash": doc_hash,
                "page_count": len(pdf),
                "title": self._extract_title(page_texts),
            }

            if self.extract_images:
                metadata["images"] = images

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

    def _build_page_text(self, page: fitz.Page, page_images: List[Dict[str, Any]]) -> str:
        """构造单页文本，并把图片占位符嵌进正文。"""
        page_text = page.get_text().strip()
        placeholders = [f"[IMAGE: {item['id']}]" for item in page_images]
        if not placeholders:
            return page_text
        if not page_text:
            return "\n".join(placeholders)
        return f"{page_text}\n\n" + "\n".join(placeholders)

    def _extract_page_image_metadata(self, page: fitz.Page, doc_hash: str, page_num: int) -> List[Dict[str, Any]]:
        """提取单页图片元数据。"""
        images: List[Dict[str, Any]] = []
        image_dir = self.image_storage_dir / doc_hash
        image_list = page.get_images(full=True)
        for image_index, image_info in enumerate(image_list, start=1):
            xref = image_info[0]
            image_id = self._build_image_id(doc_hash, page_num, image_index)
            images.append(
                {
                    "id": image_id,
                    "xref": xref,
                    "page": page_num,
                    "index": image_index,
                    "storage_dir": str(image_dir),
                    "placeholder": f"[IMAGE: {image_id}]",
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
