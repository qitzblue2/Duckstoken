"""Download the latest official llama.cpp Windows CPU build and unpack it into ./engine for bundling.

Run by the release workflow (and build_windows.bat). Only llama-server.exe and the DLLs it needs are kept.
Verifies the archive against GitHub's published SHA-256 digest when the API provides one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from duckstoken import downloads  # noqa: E402

API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"


def _get(url: str, accept: str | None = None):
    headers = {"User-Agent": "Duckstoken-build"}
    if accept:
        headers["Accept"] = accept
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120)


def pick_asset(release: dict, backend: str = "cpu") -> dict:
    assets = [{"name": a["name"], "url": a["browser_download_url"], "size": a["size"], "digest": a.get("digest") or ""}
              for a in release.get("assets", []) if a["name"].endswith(".zip")]
    win = [a for a in assets if "win" in a["name"].lower() and "x64" in a["name"].lower()
           and "arm" not in a["name"].lower()]
    picked = downloads.pick_assets(win, backend)
    if not picked:
        raise SystemExit(f"No Windows {backend} build found among: {[a['name'] for a in win]}")
    return picked[0]


def keep_only_server(folder: Path) -> list[str]:
    """Remove every .exe except llama-server.exe (the other tools only bloat the app)."""
    removed = []
    for p in folder.rglob("*.exe"):
        if p.name.lower() != "llama-server.exe":
            p.unlink()
            removed.append(p.name)
    return removed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="engine")
    ap.add_argument("--backend", default="cpu", choices=["cpu", "vulkan"])
    a = ap.parse_args()

    with _get(API, "application/vnd.github+json") as r:
        release = json.load(r)
    asset = pick_asset(release, a.backend)
    print(f"llama.cpp {release.get('tag_name')}: {asset['name']} ({asset['size'] / 2**20:.1f} MB)")

    out = Path(a.out)
    tmp = out.with_name(out.name + "_dl")
    shutil.rmtree(out, ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    zip_path = tmp / asset["name"]
    h = hashlib.sha256()
    with _get(asset["url"]) as r, open(zip_path, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            h.update(chunk)
    digest = asset["digest"].split(":", 1)[-1].lower() if asset["digest"] else ""
    if digest and digest != h.hexdigest():
        raise SystemExit(f"SHA-256 mismatch for {asset['name']}: expected {digest}, got {h.hexdigest()}")
    print("sha256", h.hexdigest(), "(verified)" if digest else "(no published digest to compare)")

    downloads.safe_extract(zip_path, out)
    shutil.rmtree(tmp, ignore_errors=True)
    removed = keep_only_server(out)
    exe = downloads.find_server_exe(out)
    if not exe:
        raise SystemExit("llama-server.exe not found in the archive")
    if exe.parent != out:  # flatten: the app looks for engine/llama-server.exe
        for item in exe.parent.iterdir():
            shutil.move(str(item), str(out / item.name))
    (out / "VERSION.txt").write_text(f"{release.get('tag_name')}\n{asset['name']}\nsha256 {h.hexdigest()}\n")
    print(f"engine ready in {out} ({sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) / 2**20:.0f} MB); "
          f"dropped {len(removed)} extra tools")


if __name__ == "__main__":
    main()
