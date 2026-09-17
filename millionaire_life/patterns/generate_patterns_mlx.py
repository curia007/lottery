#!/usr/bin/env python3
"""
generate_patterns_mlx.py

Entry point to analyze and generate numerical patterns in Millionaire Life lottery balls using MLX.
"""

import sys
from pathlib import Path

_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent.parent
for _p in [str(_project_root), str(_current_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from millionaire_life.patterns.millionaire_patterns_mlx import main
except ImportError:
    from millionaire_patterns_mlx import main

if __name__ == "__main__":
    main()
