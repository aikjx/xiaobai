"""允许 `python -m xiaobai_voice …` 直接调用。"""
from .cli import main

raise SystemExit(main())
