"""pytest 测试基础设施。

当前 `repro` 目录还没有独立安装成包，因此测试运行时先手动把 `src/`
加入 `sys.path`。后面如果补了自己的 `.venv` 和安装步骤，这里可以再简化。
"""

from __future__ import annotations

import sys
from pathlib import Path


REPRO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPRO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
