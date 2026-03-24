"""pytest 测试基础设施。

当前 `repro` 目录还没有独立安装成包，因此测试运行时先手动把 `src/`
加入 `sys.path`。后面如果补了自己的 `.venv` 和安装步骤，这里可以再简化。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPRO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPRO_ROOT / "src"
TEST_ROOT = REPRO_ROOT / "tests"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from helpers import load_test_settings, make_temp_config, unique_collection_name


@pytest.fixture()
def temp_config_path(tmp_path: Path) -> Path:
    """为当前测试生成一份临时配置文件。"""
    return make_temp_config(tmp_path)


@pytest.fixture()
def test_settings(temp_config_path: Path):
    """加载当前测试的临时配置。"""
    return load_test_settings(temp_config_path)


@pytest.fixture()
def unique_collection() -> str:
    """返回一个唯一 collection 名称，避免污染已有数据。"""
    return unique_collection_name()
