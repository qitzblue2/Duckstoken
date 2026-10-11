"""Resumable downloads (Hugging Face GGUFs, llama.cpp builds) with progress tracking."""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path

UA = "Duckstoken/0.1 (+https://github.com/qitzblue2/duckstoken)"
HF = "https://huggingface.co"
LLAMA_RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
_SHARD = re.compile(r"^(?P<base>.*)-(?P<n>\d{5})-of-(?P<total>\d{5})\.gguf$", re.I)


class Job:
    def __init__(self, title: str, kind: str):
        self.id = uuid.uuid4().hex[:10]
        self.title, self.kind = title, kind
        self.status = "queued"  # queued running done error cancelled
        self.done = self.total = 0
        self.speed = 0.0
        self.message = ""
        self.error = ""
        self.result: dict = {}
        self.cancel = threading.Event()
        self.created = time.time()

    def to_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "kind": self.kind, "status": self.status,
                "done": self.done, "total": self.total, "speed": self.speed, "message": self.message,
                "error": self.error, "result": self.result}


class JobManager:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def add(self, job: Job, target, *args) -> Job:
        with self._lock:
            self.jobs[job.id] = job

        def run():
            job.status = "running"
            try:
                target(job, *args)
                if job.status == "running":
                    job.status = "done"
            except Cancelled:
                job.status = "cancelled"
            except Exception as e:  # noqa: BLE001
                job.status, job.error = "error", str(e) or e.__class__.__name__

        threading.Thread(target=run, daemon=True, name=f"job-{job.id}").start()
        return job

    def list(self) -> list[dict]:
        with self._lock:
            return [j.to_dict() for j in sorted(self.jobs.values(), key=lambda j: -j.created)][:30]

    def cancel(self, job_id: str) -> bool:
        j = self.jobs.get(job_id)
        if j:
            j.cancel.set()
        return bool(j)


class Cancelled(Exception):
    pass


def _request(url: str, headers: dict | None = None, timeout: int = 30):
    h = {"User-Agent": UA}
    h.update(headers or {})
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout)


def _hf_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}


def download_file(job: Job, url: str, dest: Path, headers: dict | None = None,
                  base_done: int = 0, total_override: int | None = None) -> None:
    """Resumable download to dest (via dest.part). Updates job.done/total/speed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    h = dict(headers or {})
    if have:
        h["Range"] = f"bytes={have}-"
    try:
        resp = _request(url, h, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:  # already complete
            os.replace(part, dest)
            return
        raise RuntimeError(f"Download failed: HTTP {e.code} for {url}") from e
    status = getattr(resp, "status", 200)
    length = int(resp.headers.get("Content-Length") or 0)
    if status == 206:
        total = have + length
    else:
        have, total = 0, length
    if total_override and not total:
        total = total_override
    mode = "ab" if (status == 206 and have) else "wb"
    free = shutil.disk_usage(dest.parent).free
    if total and total - have > free:
        raise RuntimeError(f"Not enough disk space on {dest.drive or dest.parent}: need "
                           f"{(total - have) / 2**30:.1f} GB, have {free / 2**30:.1f} GB free")
    last_t, last_b = time.time(), have
    with open(part, mode) as f, resp:
        got = have
        while True:
            if job.cancel.is_set():
                raise Cancelled()
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            job.done = base_done + got
            now = time.time()
            if now - last_t >= 1:
                job.speed = (got - last_b) / (now - last_t)
                last_t, last_b = now, got
    if total and got < total:
        raise RuntimeError("Connection dropped before the file finished - press Download again to resume.")
    os.replace(part, dest)


# ----------------------------------------------------------- Hugging Face
def list_hf_models(repo: str, token: str = "") -> list[dict]:
    """GGUF files in a repo, grouped (split shards -> one entry)."""
    if not re.fullmatch(r"[\w.\-]+/[\w.\-]+", repo):
        raise ValueError("Repository must look like  owner/name")
    url = f"{HF}/api/models/{repo}/tree/main?recursive=true"
    with _request(url, _hf_headers(token)) as r:
        entries = json.load(r)
    groups: dict[str, dict] = {}
    for e in entries:
        if e.get("type") != "file" or not e["path"].lower().endswith(".gguf"):
            continue
        path, size = e["path"], int(e.get("size") or 0)
        m = _SHARD.match(path)
        key = m["base"] if m else path
        g = groups.setdefault(key, {"name": key if not m else key + "-*.gguf", "files": [], "size": 0})
        g["files"].append({"path": path, "size": size})
        g["size"] += size
    out = []
    for g in groups.values():
        g["files"].sort(key=lambda f: f["path"])
        low = g["name"].lower()
        g["kind"] = ("draft" if ("eagle" in low or "draft" in low) else
                     "vision" if "mmproj" in low else "model")
        g["size_gb"] = round(g["size"] / 2**30, 2)
        out.append(g)
    out.sort(key=lambda g: (g["kind"] != "model", -g["size"]))
    return out


def start_hf_download(jobs: JobManager, repo: str, files: list[dict], dest_dir: Path, token: str) -> Job:
    if not re.fullmatch(r"[\w.\-]+/[\w.\-]+", repo) or ".." in repo.split("/"):
        raise ValueError("Repository must look like  owner/name")
    safe_repo = repo.replace("/", "__")
    job = Job(f"{repo}: {files[0]['path']}" + (f" (+{len(files) - 1} parts)" if len(files) > 1 else ""),
              "model")
    job.total = sum(f["size"] for f in files)

    def run(job: Job):
        base = 0
        for f in files:
            name = os.path.basename(f["path"])
            if not name.lower().endswith(".gguf") or "/" in name or "\\" in name:
                raise RuntimeError(f"Refusing suspicious file name: {name}")
            dest = dest_dir / safe_repo / name
            job.message = f"Downloading {name}"
            if dest.exists() and f["size"] and dest.stat().st_size == f["size"]:
                base += f["size"]
                job.done = base
                continue
            url = f"{HF}/{repo}/resolve/main/{urllib.parse.quote(f['path'])}"
            download_file(job, url, dest, _hf_headers(token), base_done=base)
            base += f["size"]
        job.message = "Finished"
        job.result = {"path": str(dest_dir / safe_repo / os.path.basename(files[0]["path"]))}

    return jobs.add(job, run)


# --------------------------------------------------------------- llama.cpp
def list_llama_assets(backend: str = "auto") -> dict:
    with _request(LLAMA_RELEASES, {"Accept": "application/vnd.github+json"}) as r:
        rel = json.load(r)
    assets = [{"name": a["name"], "url": a["browser_download_url"], "size": a["size"]}
              for a in rel.get("assets", []) if a["name"].endswith(".zip")]
    win = [a for a in assets if "win" in a["name"].lower() and "x64" in a["name"].lower()
           and "arm" not in a["name"].lower()]
    return {"tag": rel.get("tag_name", ""), "assets": win, "recommended": pick_assets(win, backend)}


def pick_assets(win_assets: list[dict], backend: str) -> list[dict]:
    """Choose the zip(s) to install for a backend. CUDA needs the cudart zip too."""
    def has(a, *words):
        n = a["name"].lower()
        return all(w in n for w in words)

    if backend == "cuda":
        main = [a for a in win_assets if has(a, "cuda") and not has(a, "cudart")]
        # prefer CUDA 12.x (still supports Maxwell/Pascal cards); newest 12.x first
        main.sort(key=lambda a: (not re.search(r"cuda-12", a["name"]), a["name"]), reverse=False)
        rt = [a for a in win_assets if has(a, "cudart")]
        if main:
            pick = main[0]
            ver = re.search(r"cuda-([\d.]+)", pick["name"])
            rts = [a for a in rt if ver and ver.group(1) in a["name"]] or rt
            return [pick] + rts[:1]
    elif backend == "vulkan":
        v = [a for a in win_assets if has(a, "vulkan")]
        if v:
            return v[:1]
    cpu = [a for a in win_assets if (has(a, "cpu") or has(a, "avx2")) and not has(a, "cuda")
           and not has(a, "vulkan") and not has(a, "hip") and not has(a, "sycl") and not has(a, "opencl")]
    cpu.sort(key=lambda a: ("avx2" not in a["name"].lower() and "cpu" not in a["name"].lower(), a["name"]))
    return cpu[:1]


def safe_extract(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            target = (dest / m.filename).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"Unsafe path in zip: {m.filename}")
        z.extractall(dest)


def find_server_exe(folder: Path) -> Path | None:
    names = ("llama-server.exe", "llama-server")
    for p in folder.rglob("*"):
        if p.is_file() and p.name in names:
            return p
    return None


def start_llama_install(jobs: JobManager, assets: list[dict], tag: str, backend: str,
                        engines_dir: Path, on_done) -> Job:
    job = Job(f"llama.cpp {tag} ({backend})", "engine")
    job.total = sum(a.get("size", 0) for a in assets)

    def run(job: Job):
        dest = engines_dir / f"{tag}-{backend}"
        base = 0
        for a in assets:
            if urllib.parse.urlparse(a["url"]).hostname not in ("github.com", "objects.githubusercontent.com",
                                                                  "release-assets.githubusercontent.com"):
                raise RuntimeError("Refusing to download engine from an unexpected host")
            zp = engines_dir / "_downloads" / a["name"]
            job.message = f"Downloading {a['name']}"
            download_file(job, a["url"], zp, base_done=base, total_override=a.get("size"))
            base += a.get("size", 0)
            job.message = f"Extracting {a['name']}"
            safe_extract(zp, dest)
            zp.unlink(missing_ok=True)
        exe = find_server_exe(dest)
        if not exe:
            raise RuntimeError("Download finished but llama-server was not found inside the archive")
        job.result = {"path": str(exe)}
        job.message = "Installed"
        on_done(str(exe))

    return jobs.add(job, run)


def installed_engines(engines_dir: Path) -> list[dict]:
    out = []
    if engines_dir.is_dir():
        for d in sorted(engines_dir.iterdir()):
            if d.is_dir() and not d.name.startswith("_"):
                exe = find_server_exe(d)
                if exe:
                    out.append({"name": d.name, "path": str(exe)})
    return out
