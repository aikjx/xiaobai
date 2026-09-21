"""Xiaobai CLI: serve / desktop / download / selftest / license."""
from __future__ import annotations

import io
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 运行期资源根目录（frozen 时由 main() 解析；开发态为 None，loader 使用 __file__）
# config/loader.py 等模块可以读取此变量作为打包后资源的首选路径。
# ---------------------------------------------------------------------------
_RESOURCE_ROOT: Path | None = None


def _ensure_windowed_streams() -> None:
    if sys.stdout is None:
        sys.stdout = io.StringIO()
    if sys.stderr is None:
        try:
            sys.stderr = _open_diagnostic_stderr()
        except Exception:  # noqa: BLE001
            sys.stderr = io.StringIO()


def _open_diagnostic_stderr():
    from platform import system
    s = system()
    if s == "Windows":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        root = Path(base) / "mox" / "xiaobai" / "logs"
    elif s == "Darwin":
        root = Path.home() / "Library" / "Logs" / "mox" / "xiaobai"
    else:
        xdg = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
        root = Path(xdg) / "mox" / "xiaobai" / "logs"
    root.mkdir(parents=True, exist_ok=True)
    p = root / f"windowed-{time.strftime('%Y%m%d-%H%M%S')}.log"
    fp = open(p, "a", encoding="utf-8", buffering=1)
    bad = ["jianpu-ly: ", "music21: WARNING", "music21: INFO"]

    class W:
        def write(self_self, data):
            if not data:
                return 0
            if any(x in data for x in bad):
                return 0
            try:
                return fp.write(data)
            except Exception:  # noqa: BLE001
                return 0
        def flush(self_self):
            try:
                fp.flush()
            except Exception:  # noqa: BLE001
                pass
        def isatty(self_self):
            return False
    return W()


def _path_append(p: str) -> None:
    paths = os.environ.get("PATH", "").split(os.pathsep)
    if p not in paths:
        os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")


def _sitepkgs_under_venv(venv: Path):
    out = []
    win = venv / "Lib" / "site-packages"
    nix = venv / "lib" / ("python%d.%d" % (sys.version_info.major, sys.version_info.minor)) / "site-packages"
    for x in (win, nix):
        if x.is_dir():
            out.append(str(x))
    return out


def _versioned_sitepkgs_ok(p: Path) -> bool:
    """site-packages 目录必须属于当前主版本 Python（避免 3.12 解释器加载 3.8 site-packages → typing/pydantic ABI 崩溃）。"""
    try:
        s = str(p.resolve()).lower()
    except Exception:
        s = str(p).lower()
    target = "python%d%d" % (sys.version_info.major, sys.version_info.minor)
    # 直接命中版本号片段（如 `Python312`、`python3.12`、`lib/python3.12`）
    import re
    loose = r"python[._\- ]?%d[._\- ]?%d" % (sys.version_info.major, sys.version_info.minor)
    if re.search(loose, s):
        return True
    # 以 .../Lib/site-packages 结尾 → 检查其父链是否含当前版本前缀（当前 sys.prefix 所在路径无条件接受）。
    try:
        cur = Path(sys.prefix).resolve().lower()
        if s.startswith(cur):
            return True
    except Exception:
        pass
    # 没有明确版本标记的通用路径（比如 ~/.mox/models/voice、自建 venv），不允许混用：
    # 仅当父目录包含 PythonXX 子目录匹配当前版本，或显式 `.venv/pyvenv.cfg` 指定版本时才放行。
    parent = p.parent
    while parent != parent.parent:
        pname = parent.name.lower()
        if re.match(r"^python\d+$", pname) and pname != target.lower():
            return False  # 明确其他版本（如 Python38）→ 拒绝
        pyvenv = parent / "pyvenv.cfg"
        if pyvenv.is_file():
            try:
                cfg_text = pyvenv.read_text(encoding="utf-8", errors="ignore")
                if ("version_info = %d.%d" % (sys.version_info.major, sys.version_info.minor)) in cfg_text.replace(" ", ""):
                    return True
                # home= 行若包含 pythonXX 并且匹配当前解释器目录则放行
            except Exception:
                pass
            # 任何其他 venv 一律拒绝，避免跨 ABI 混用。
            return False
        parent = parent.parent
    return False


def _discover_and_inject_sitepkgs() -> list:
    import site as _site
    hits: list = []
    explicit = os.environ.get("XIAOBAI_VENV_DIR")
    if explicit and Path(explicit).is_dir():
        hits.extend(_sitepkgs_under_venv(Path(explicit)))
    try:
        for s in _site.getsitepackages():
            if Path(s).is_dir():
                hits.append(s)
    except Exception:  # noqa: BLE001
        pass
    # 当前解释器所在的 DLLs 目录一定兼容（仅注入同版本）；禁止混编注入其他 Python 版本 DLLs。
    current_prefix = Path(getattr(sys, "prefix", sys.executable and os.path.dirname(sys.executable) or ""))
    for sub in ("DLLs",):
        cand = current_prefix / sub
        if cand.is_dir():
            hits.append(str(cand))
    if os.name == "nt":
        try:
            import subprocess
            res = subprocess.run(["where.exe", "python"], capture_output=True, text=True, timeout=6)
            for line in (res.stdout or "").splitlines():
                exe = line.strip('" ').strip()
                if not exe or not Path(exe).is_file():
                    continue
                # 跳过当前解释器自身：DLLs 已在上方单独加入；避免循环。
                try:
                    same = Path(exe).resolve() == Path(sys.executable).resolve()
                except Exception:
                    same = False
                try:
                    r = subprocess.run(
                        [exe, "-c", "import site,sys; print('|'.join(site.getsitepackages()+[sys.prefix]))"],
                        capture_output=True, text=True, timeout=6,
                    )
                    if r.returncode or not r.stdout.strip():
                        continue
                    for p in r.stdout.strip().split("|"):
                        pp = Path(p)
                        if pp.is_dir() and "site-packages" in p:
                            hits.append(p)
                        elif pp.is_dir() and same:
                            # 仅当前解释器自身的 prefix 允许注入 DLLs（site-packages 下面已统一追加）。
                            hits.append(str(pp / "Lib/site-packages"))
                except Exception:  # noqa: BLE001
                    continue
        except Exception:  # noqa: BLE001
            pass
    home = Path.home()
    for cand in [home / "miniconda3" / "envs", home / "anaconda3" / "envs", home / "miniforge3" / "envs",
                 Path(os.environ.get("PROGRAMDATA") or r"C:\ProgramData") / "miniconda3" / "envs"]:
        try:
            if cand.is_dir():
                for env in cand.iterdir():
                    sp = env / "Lib" / "site-packages"
                    if sp.is_dir():
                        hits.append(str(sp))
                    # 不注入 env/DLLs：避免跨版本解释器 ABI 冲突（_socket / _ssl 等混版加载会直接崩溃）。
        except Exception:  # noqa: BLE001
            pass
    uniq = []
    for h in hits:
        try:
            pp = Path(h)
            # DLLs 目录不属于 site-packages；只要是当前解释器 prefix 下的或显式 current_prefix/DLLs 就放行。
            name = pp.name.lower()
            if name == "dlls" or name == "scripts":
                try:
                    under_cur = Path(sys.prefix).resolve() in pp.resolve().parents
                except Exception:
                    under_cur = False
                if not under_cur and pp.resolve() != (current_prefix / name).resolve():
                    continue
            elif "site-packages" in str(pp).lower():
                if not _versioned_sitepkgs_ok(pp):
                    continue
        except Exception:
            pass
        norm = str(Path(h).resolve()) if Path(h).exists() else str(h)
        if norm not in uniq:
            uniq.append(norm)
    for h in reversed(uniq):
        if h and h not in sys.path and Path(h).is_dir():
            sys.path.insert(0, h)
    return uniq


def _inject_dll_dirs() -> None:
    roots: list[Path] = []
    if getattr(sys, "frozen", False):
        roots.append(Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)))
        roots.append(Path(sys.executable).parent)
    for sp in _discover_and_inject_sitepkgs():
        spp = Path(sp)
        roots.append(spp)
        # 仅当 site-packages 父目录是当前解释器时，才追加其 DLLs/Scripts；
        # 其它环境 site-packages 只用来找 numpy/.libs / torch/lib 这类第三方包独立 DLL。
        try:
            prefix_cand = spp.parent.parent
            same_prefix = Path(sys.prefix).resolve() == prefix_cand.resolve()
        except Exception:
            same_prefix = False
        if same_prefix:
            roots.append(spp.parent.parent / "DLLs")
            roots.append(spp.parent.parent / "Scripts")
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for suffix in ("", "numpy/.libs", "onnxruntime/capi",
                       "PySide6/Qt6/bin", "PySide6/plugins", "shiboken6",
                       "_sounddevice_data", "sounddevice",
                       "torch/lib", "torch/bin",
                       "nvidia/cudnn_runtime/bin", "nvidia/cuda_runtime/bin"):
            c = root / suffix if suffix else root
            try:
                if not c.is_dir():
                    continue
                key = str(c.resolve()).lower()
                if key in seen:
                    continue
                seen.add(key)
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(str(c))  # type: ignore[attr-defined]
                    except Exception:  # noqa: BLE001
                        _path_append(str(c))
                else:
                    _path_append(str(c))
            except Exception:  # noqa: BLE001
                pass


def _inject_env() -> None:
    home_voice = str(Path.home() / ".mox" / "models" / "voice")
    for k, v in {
        "FISH_SPEECH_CKPT_DIR": home_voice,
        "COSYVOICE_CKPT_DIR": home_voice,
        "MOX_VOICE_PORT": "30010",
    }.items():
        os.environ.setdefault(k, v)


def _setup_logging() -> None:
    import logging
    from .config.loader import default_log_path
    lv = getattr(logging, str(os.environ.get("XIAOBAI_LOG_LEVEL") or "INFO").upper(), logging.INFO)
    log_dir = default_log_path()
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(lv)
    if any(getattr(h, "_xiaobai_", False) for h in root.handlers):
        return
    try:
        fh = logging.FileHandler(log_dir / ("xiaobai-%s.log" % time.strftime("%Y%m%d")), encoding="utf-8")
        fh.setFormatter(fmt)
        fh._xiaobai_ = True  # type: ignore[attr-defined]
        root.addHandler(fh)
        if hasattr(sys.stderr, "write") and not isinstance(sys.stderr, io.StringIO):
            sh = logging.StreamHandler(sys.stderr)
            sh.setFormatter(fmt)
            sh._xiaobai_ = True  # type: ignore[attr-defined]
            root.addHandler(sh)
    except Exception:  # noqa: BLE001
        pass


def _progress(ev):
    if ev.get("state") in ("cached", "done"):
        return
    print("\r  · %.1f%%  %.2f MB/s  ETA %.0fs  " % (
        float(ev.get("progress_pct") or 0.0),
        float(ev.get("speed_mbps") or 0.0),
        float(ev.get("eta_s") or 0.0),
    ), end="", flush=True)


def _spawn_serve(port: int) -> None:
    import subprocess
    if getattr(sys, "frozen", False):
        args = [sys.executable, "serve", "--port", str(port)]
    else:
        args = [sys.executable, "-m", "xiaobai_voice", "serve", "--port", str(port)]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         stdin=subprocess.DEVNULL, creationflags=flags, close_fds=True)
    except Exception as exc:  # noqa: BLE001
        import logging
        logging.getLogger("xiaobai.cli").warning("spawn serve failed: %s", exc)


def _ast_scan_fish():
    import ast
    root = Path(__file__).resolve().parent
    issues = []
    for p in root.rglob("*.py"):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    if a.name.startswith("fish_speech"):
                        issues.append("%s:%s import %s" % (p, getattr(n, "lineno", "?"), a.name))
            elif isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("fish_speech"):
                issues.append("%s:%s from %s import ..." % (p, getattr(n, "lineno", "?"), n.module))
    return issues


def main(argv=None) -> int:
    global _RESOURCE_ROOT
    _ensure_windowed_streams()

    # --- frozen 模式下资源根目录解析（onedir: exe.parent 或 _MEIPASS） ----
    if getattr(sys, "frozen", False):
        _log = logging.getLogger("xiaobai.cli")
        exe_parent = Path(sys.executable).resolve().parent
        meipass = Path(getattr(sys, "_MEIPASS", exe_parent))
        try:
            _log.info("frozen: sys.executable = %s", sys.executable)
            _log.info("frozen: sys._MEIPASS    = %s", meipass)
        except Exception:  # logging 尚未初始化时兜底到 stderr
            print("[frozen] sys.executable =", sys.executable, file=sys.stderr)
            print("[frozen] sys._MEIPASS    =", meipass, file=sys.stderr)

        # PyInstaller 6+ onedir: datas 出现在 dist/Xiaobai/_internal/ 或根下
        # 优先 exe.parent/xiaobai_voice/config，回退 _MEIPASS/xiaobai_voice/config
        cand1 = exe_parent / "xiaobai_voice" / "config"
        cand2 = meipass / "xiaobai_voice" / "config"
        if cand1.is_dir():
            _RESOURCE_ROOT = exe_parent
        elif cand2.is_dir():
            _RESOURCE_ROOT = meipass
        else:
            # 都不存在时仍取 _MEIPASS 作为兜底（后续读文件会各自处理 FileNotFound）
            _RESOURCE_ROOT = meipass
            try:
                _log.warning(
                    "frozen: xiaobai_voice/config 目录既不在 %s 也不在 %s，"
                    "将以 _MEIPASS 作为 _RESOURCE_ROOT 兜底。",
                    cand1, cand2,
                )
            except Exception:
                print("[frozen] WARNING: config dir missing, fallback _MEIPASS =", meipass, file=sys.stderr)

    _inject_dll_dirs()
    _inject_env()
    _setup_logging()
    import argparse
    ap = argparse.ArgumentParser(prog="xiaobai", description="Xiaobai voice + desktop")
    s = ap.add_subparsers(dest="cmd", required=True)

    p = s.add_parser("serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=int(os.environ.get("MOX_VOICE_PORT") or 30010))
    p.add_argument("--log-level", default="info", choices=["trace", "debug", "info", "warning", "error"])

    p = s.add_parser("desktop")
    p.add_argument("--skip-serve", action="store_true")
    p.add_argument("--port", type=int, default=int(os.environ.get("MOX_VOICE_PORT") or 30010))

    p = s.add_parser("download")
    p.add_argument("--model-id", default=None)
    p.add_argument("--defaults", action="store_true")

    p = s.add_parser("selftest")
    p.add_argument("--full", action="store_true")

    s.add_parser("license")

    args = ap.parse_args(argv)

    if args.cmd == "serve":
        from .service.main import run_server
        run_server(host=args.host, port=args.port, log_level=args.log_level)
        return 0

    if args.cmd == "desktop":
        if not args.skip_serve:
            _spawn_serve(args.port)
        from .desktop.app import run_desktop
        return run_desktop(args) or 0

    if args.cmd == "download":
        from .models import ModelDownloader, ModelRegistry
        reg = ModelRegistry()
        dl = ModelDownloader(reg)
        targets = []
        if args.defaults:
            from .config.loader import ConfigLoader
            tier = str((ConfigLoader().get("voice.license_tier") or "auto")).lower()
            for m in reg.list_all():
                if m.optional:
                    continue
                if m.id == "tts-fish-s2-pro" and tier == "apache2":
                    continue
                targets.append(m.id)
        elif args.model_id:
            targets.append(args.model_id)
        if not targets:
            ap.error("需要 --model-id <id> 或 --defaults")
        for mid in targets:
            t0 = time.time()
            print("[download] %s 开始…" % mid, flush=True)
            try:
                path = dl.download(mid, on_progress=_progress)
                print("\n[download] %s OK · %s · %.1fs" % (mid, path, time.time() - t0), flush=True)
            except Exception as exc:  # noqa: BLE001
                print("\n[download] %s FAILED: %s" % (mid, exc), flush=True)
                return 2
        return 0

    if args.cmd == "selftest":
        from .tests.selftest import run_selftest
        return run_selftest(full=args.full) or 0

    if args.cmd == "license":
        from .config.loader import ConfigLoader
        tier = str((ConfigLoader().get("voice.license_tier") or "auto")).lower()
        print("[license] tier =", tier)
        allowed = ["browser"]
        if tier in ("auto", "research"):
            allowed += ["fish_s2", "cosyvoice2"]
        else:
            allowed += ["cosyvoice2"]
        print("[license] allowed tts engines =", allowed)
        if tier == "apache2":
            issues = _ast_scan_fish()
            if issues:
                print("[license] AST 命中 fish_speech:")
                for it in issues:
                    print("  -", it)
                return 3
            print("[license] AST 扫描通过：0 条 fish_speech import 语句。")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
