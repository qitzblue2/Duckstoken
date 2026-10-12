import argparse
import os

from . import __version__


def main() -> None:
    ap = argparse.ArgumentParser(prog="duckstoken", description="Run big local LLMs on modest PCs.")
    ap.add_argument("--no-browser", action="store_true", help="don't open any window (same as --headless)")
    ap.add_argument("--headless", action="store_true", help="serve the UI only; no window, tray or browser")
    ap.add_argument("--minimized", action="store_true", help="start hidden in the system tray")
    ap.add_argument("--port", type=int, help="UI port (default 8765)")
    ap.add_argument("--data-dir", help="where settings/engines are stored")
    ap.add_argument("--selftest", metavar="FILE", help="check that this build contains everything it needs, write a JSON report to FILE and exit")
    ap.add_argument("--version", action="version", version=f"duckstoken {__version__}")
    a = ap.parse_args()
    if a.selftest:
        from . import selftest
        raise SystemExit(selftest.run(a.selftest))
    if a.data_dir:
        os.environ["DUCKSTOKEN_HOME"] = a.data_dir
    from .app import run
    run(open_browser=False if a.no_browser else None, port=a.port, headless=a.headless or a.no_browser,
        minimized=a.minimized)


if __name__ == "__main__":
    main()
