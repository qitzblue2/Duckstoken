#!/usr/bin/env python3
"""Build the Windows app with PyInstaller. One place for the flags so the portable exe, the installer payload and
build_windows.bat can never drift apart.

    python tools/build_exe.py onefile            # dist/Duckstoken.exe  (portable)
    python tools/build_exe.py onedir --distpath dist-installer   # dist-installer/Duckstoken/ (installer payload)

`webview` is imported by name at run time (duckstoken/shell.py), so PyInstaller cannot see it: it must be collected explicitly
or the packaged app silently falls back to the browser. `python Duckstoken.exe --selftest out.json` verifies the result.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

COLLECT = ["webview", "clr_loader", "pythonnet"]      # pywebview + its Windows .NET bridge (code, data and DLLs)
HIDDEN = ["pystray._win32", "clr"]


def command(mode: str, distpath: str | None = None) -> list[str]:
    if mode not in ("onefile", "onedir"):
        raise ValueError("mode must be onefile or onedir")
    sep = os.pathsep
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", f"--{mode}", "--windowed", "--name", "Duckstoken",
           "--icon", str(Path("assets") / "duckstoken.ico"),
           "--add-data", f"duckstoken/web{sep}duckstoken/web", "--add-data", f"engine{sep}engine"]
    for name in COLLECT:
        cmd += ["--collect-all", name]
    for name in HIDDEN:
        cmd += ["--hidden-import", name]
    if distpath:
        cmd += ["--distpath", distpath]
    return cmd + ["duckstoken_launcher.py"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["onefile", "onedir"])
    ap.add_argument("--distpath")
    ap.add_argument("--dry-run", action="store_true", help="print the command and exit")
    a = ap.parse_args()
    cmd = command(a.mode, a.distpath)
    print(" ".join(cmd), flush=True)
    return 0 if a.dry_run else subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
