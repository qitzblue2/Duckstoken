import argparse
import os

from . import __version__


def main() -> None:
    ap = argparse.ArgumentParser(prog="duckstoken", description="Run big local LLMs on modest PCs.")
    ap.add_argument("--no-browser", action="store_true", help="don't open the UI automatically")
    ap.add_argument("--port", type=int, help="UI port (default 8765)")
    ap.add_argument("--data-dir", help="where settings/engines are stored")
    ap.add_argument("--version", action="version", version=f"duckstoken {__version__}")
    a = ap.parse_args()
    if a.data_dir:
        os.environ["DUCKSTOKEN_HOME"] = a.data_dir
    from .app import run
    run(open_browser=False if a.no_browser else None, port=a.port)


if __name__ == "__main__":
    main()
