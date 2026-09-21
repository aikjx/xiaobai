import sys
import os

# 允许不 pip install -e 就能运行：xiaobai_voice/ 所在目录加 sys.path
_HERE = os.path.abspath(os.path.dirname(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from xiaobai_voice.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
