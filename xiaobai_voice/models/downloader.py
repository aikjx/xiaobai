"""模型注册表 + 下载器（httpx Range + SHA256 + 3 次指数退避 + SSE 进度）。"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tarfile
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import httpx
import yaml
from tqdm import tqdm

from ..config.loader import default_voice_models_dirs

log = logging.getLogger("xiaobai.models")

MODELS_YAML_NAME = "models.yaml"


@dataclass
class ModelStatus:
    id: str
    name: str
    license: str
    size_mb: float
    downloaded: bool
    sha256_ok: bool | None
    local_root: str | None
    engine: str | None = None
    category: str | None = None
    optional: bool = False


class ModelRegistry:
    """读 config/models.yaml，按优先级解析模型实际路径。"""

    def __init__(self, models_yaml: Path | None = None, extra_dirs: Iterable[Path] | None = None) -> None:
        if models_yaml is None:
            models_yaml = Path(__file__).resolve().parent / ".." / "config" / MODELS_YAML_NAME
        self.models_yaml = Path(models_yaml)
        self.extra_dirs = [Path(d) for d in (extra_dirs or [])]
        with open(self.models_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self.version = int(data.get("version") or 1)
        self.models_raw: list[dict] = list(data.get("models") or [])

    # ------------------------------------------------------------------ paths
    def model_root_candidates(self, subdir: str) -> list[Path]:
        out: list[Path] = []
        out.extend(self.extra_dirs)
        out.extend(default_voice_models_dirs())
        return [Path(p) / subdir for p in out]

    def find_local_root(self, model_id: str) -> Path | None:
        meta = self.meta(model_id)
        if not meta:
            return None
        subdir = meta["subdir"]
        for root in self.model_root_candidates(subdir):
            if self._check_entry(root, meta):
                return root
        return None

    def _check_entry(self, root: Path, meta: dict) -> bool:
        if not root.is_dir():
            return False
        entry = meta.get("entry") or {}
        if not entry:
            # CosyVoice2 等 entry 为空 → 只要目录存在就算通过
            return True
        for k, v in entry.items():
            if v in ("", None):
                continue
            p = root / v
            if not p.is_file():
                return False
        return True

    # ------------------------------------------------------------------ meta
    def meta(self, model_id: str) -> dict | None:
        for m in self.models_raw:
            if m.get("id") == model_id:
                return m
        return None

    def list_all(self) -> list[ModelStatus]:
        out: list[ModelStatus] = []
        for m in self.models_raw:
            local = self.find_local_root(m["id"])
            out.append(
                ModelStatus(
                    id=m["id"],
                    name=m.get("name", m["id"]),
                    license=m.get("license", "Unknown"),
                    size_mb=float(m.get("size_mb") or 0),
                    downloaded=local is not None,
                    sha256_ok=self._verify_sha256(m, local) if local else None,
                    local_root=str(local) if local else None,
                    engine=m.get("engine"),
                    category=m.get("category"),
                    optional=bool(m.get("optional")),
                )
            )
        return out

    def resolve(self, model_id: str) -> dict | None:
        """返回 {root, entry} 供后端路径解析用。"""
        meta = self.meta(model_id)
        if not meta:
            return None
        local = self.find_local_root(model_id)
        if local is None:
            return None
        return {
            "id": model_id,
            "root": str(local),
            "entry": dict(meta.get("entry") or {}),
        }

    def _verify_sha256(self, meta: dict, root: Path | None) -> bool | None:
        expected = str(meta.get("sha256") or "").strip().lower()
        if not expected or expected.startswith("tbd"):
            return None
        # 校验"原始下载包"，不在每个展开子文件上跑；下载阶段已校验写入完成的 .tar.gz/.pt
        # 如果用户手动下载解压到 models/，这里找不到源包 → 返回 None（未知）
        subdir = meta["subdir"]
        archive_format = meta.get("archive_format") or "tgz"
        if archive_format == "file":
            names = [os.path.basename(url) for url in (meta.get("urls") or []) if url]
            for n in names:
                pkg = (root or Path()).parent / n
                if pkg.is_file():
                    return _sha256_file(pkg) == expected
            return None
        # tgz：打包名固定 {subdir}.tar.gz
        pkg = (root or Path()).parent / f"{subdir}.tar.gz"
        if pkg.is_file():
            return _sha256_file(pkg) == expected
        return None


# ================================================================= downloader


class ModelDownloader:
    """下载器：Range 断点续传 + 3 次指数退避 + SHA256 校验 + 解压 tgz；

    进度通过回调 `on_progress({progress_pct, speed_mbps, eta_s, state, model_id})` 广播。
    """

    def __init__(
        self,
        registry: ModelRegistry,
        preferred_root: Path | None = None,
        user_agent: str = "xiaobai-voice/0.1",
    ) -> None:
        self.registry = registry
        self.preferred_root = Path(preferred_root) if preferred_root else default_voice_models_dirs()[1]
        self.preferred_root.mkdir(parents=True, exist_ok=True)
        self.user_agent = user_agent
        self._tasks: dict[str, dict] = {}
        self._lock = threading.Lock()

    # -------------------------------------------------------------- interface
    def download(
        self,
        model_id: str,
        *,
        on_progress: Callable[[dict], None] | None = None,
        force: bool = False,
    ) -> Path:
        """同步下载 + 校验 + 解压；返回解压后模型目录。"""
        meta = self.registry.meta(model_id)
        if not meta:
            raise ValueError(f"未知 model_id: {model_id}")
        if not force:
            local = self.registry.find_local_root(model_id)
            if local is not None:
                sha_ok = self.registry._verify_sha256(meta, local)
                if sha_ok in (True, None):
                    if on_progress:
                        on_progress(dict(model_id=model_id, state="cached", progress_pct=100.0, speed_mbps=0.0, eta_s=0))
                    return local
        urls = list(meta.get("urls") or [])
        if not urls:
            raise RuntimeError(f"模型 {model_id} 未配置任何下载 URL")
        archive_format = meta.get("archive_format") or "tgz"
        subdir = meta["subdir"]
        pkg_name = _derive_pkg_name(meta, archive_format)
        target_pkg = self.preferred_root / pkg_name
        last_exc: Exception | None = None
        for url in urls:
            try:
                self._download_one(
                    url,
                    target_pkg,
                    meta,
                    on_progress=on_progress,
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log.warning("URL %s 下载失败：%s。尝试下一源。", url, exc)
                continue
        if not target_pkg.is_file():
            raise last_exc or RuntimeError(f"模型 {model_id} 所有 URL 下载失败")
        # SHA256 失败 → 自动回删
        expected = str(meta.get("sha256") or "").strip().lower()
        if expected and not expected.startswith("tbd"):
            actual = _sha256_file(target_pkg)
            if actual != expected:
                target_pkg.unlink(missing_ok=True)
                raise RuntimeError(
                    f"SHA256 不匹配，已删除 {target_pkg.name}。"
                    f"expected={expected[:12]}… actual={actual[:12]}…"
                )
        # 解压 tgz / 移动 file
        out_dir = self.preferred_root / subdir
        if archive_format == "file":
            out_dir.mkdir(parents=True, exist_ok=True)
            # 文件型权重：复制到 {subdir}/ 下，entry 文件名固定
            target_file = out_dir / pkg_name
            if force or not target_file.is_file() or target_file.stat().st_size != target_pkg.stat().st_size:
                shutil.copy2(target_pkg, target_file)
            # 若 entry 有 ckpt 名不同，软复制（重命名会破坏 pkg 校验）
            entry: dict = meta.get("entry") or {}
            ckpt_target = entry.get("ckpt") or pkg_name
            if ckpt_target != pkg_name:
                dst = out_dir / ckpt_target
                if not dst.is_file():
                    shutil.copy2(target_pkg, dst)
        else:  # tgz
            tmpdir = Path(tempfile.mkdtemp(prefix=f"{subdir}.", dir=str(self.preferred_root)))
            try:
                with tarfile.open(target_pkg, "r:gz") as tf:
                    tf.extractall(tmpdir)
                # 处理模型包里自带顶层目录名与 subdir 不一致
                existing_children = [p for p in tmpdir.iterdir() if p.is_dir()]
                if len(existing_children) == 1:
                    extracted_root = existing_children[0]
                else:
                    extracted_root = tmpdir
                if out_dir.exists() and force:
                    shutil.rmtree(out_dir, ignore_errors=True)
                out_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(extracted_root, out_dir, dirs_exist_ok=True)
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        if on_progress:
            on_progress(dict(model_id=model_id, state="done", progress_pct=100.0, speed_mbps=0.0, eta_s=0))
        return out_dir

    # --------------------------------------------------------------- internal
    def _download_one(
        self,
        url: str,
        target: Path,
        meta: dict,
        *,
        on_progress: Callable[[dict], None] | None,
    ) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        headers = {"User-Agent": self.user_agent}
        retries = 3
        backoff = 1.0
        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            resume_from = target.stat().st_size if target.is_file() else 0
            total_size_mb = float(meta.get("size_mb") or 0)
            total_size = int(total_size_mb * 1024 * 1024) if total_size_mb else None
            if resume_from and total_size and resume_from >= total_size:
                return  # 已下完
            req_headers = dict(headers)
            if resume_from:
                req_headers["Range"] = f"bytes={resume_from}-"
            try:
                with httpx.stream(
                    "GET",
                    url,
                    headers=req_headers,
                    follow_redirects=True,
                    timeout=httpx.Timeout(30.0, connect=30.0, read=600.0),
                ) as r:
                    if r.status_code == 416:
                        return  # Range 不满足 → 认为服务器上包更小或已下完
                    if r.status_code >= 400:
                        raise httpx.HTTPStatusError(
                            f"HTTP {r.status_code}",
                            request=r.request,
                            response=r,
                        )
                    # 远端支持 Range → 206；不支持 → 200 且从头写
                    content_range = r.headers.get("Content-Range", "")
                    start_byte = resume_from
                    if content_range:
                        # bytes start-end/total
                        _, rhs = content_range.split(" ", 1)
                        rng, _total = rhs.split("/", 1)
                        start_byte = int(rng.split("-")[0])
                    total_bytes_server = int(r.headers.get("Content-Length") or 0) or None
                    mode = "ab" if start_byte > 0 else "wb"
                    downloaded = start_byte
                    t0 = time.time()
                    pbar = tqdm(
                        total=total_size or (total_bytes_server + start_byte if total_bytes_server else None),
                        initial=downloaded,
                        unit="B",
                        unit_scale=True,
                        desc=target.name,
                        disable=None,
                    )
                    last_reported = 0.0
                    with open(target, mode) as f:
                        for chunk in r.iter_bytes(1024 * 256):
                            if not chunk:
                                continue
                            f.write(chunk)
                            downloaded += len(chunk)
                            pbar.update(len(chunk))
                            now = time.time()
                            if on_progress and (now - last_reported) >= 0.5:
                                last_reported = now
                                elapsed = max(1e-6, now - t0)
                                speed = downloaded / elapsed
                                pct_est = (
                                    100.0 * downloaded / total_size
                                    if total_size
                                    else (100.0 * downloaded / (total_bytes_server or 1) if total_bytes_server else 0.0)
                                )
                                eta = (
                                    max(0.0, ((total_size or downloaded) - downloaded) / speed)
                                    if speed > 0
                                    else 0.0
                                )
                                on_progress(
                                    dict(
                                        model_id=meta["id"],
                                        state="downloading",
                                        progress_pct=min(100.0, pct_est),
                                        speed_mbps=speed / 1024 / 1024,
                                        eta_s=eta,
                                        attempt=attempt,
                                    )
                                )
                    pbar.close()
                return
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                log.warning("下载 %s 第 %d 次失败: %s", target.name, attempt, exc)
                time.sleep(backoff)
                backoff *= 2.0
        raise RuntimeError(f"下载 {target.name} 最终失败：{last_err}")


# =================================================================== helpers


def _derive_pkg_name(meta: dict, archive_format: str) -> str:
    subdir = meta["subdir"]
    if archive_format == "file":
        for url in meta.get("urls") or []:
            base = os.path.basename(url.split("?", 1)[0])
            if base:
                return base
        return f"{subdir}.pt"
    return f"{subdir}.tar.gz"


def _sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest().lower()
