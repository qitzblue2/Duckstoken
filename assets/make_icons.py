"""Regenerates the committed duck icons from the drawing in duckstoken/shell.py (needs Pillow).

    python assets/make_icons.py            rewrites assets/duckstoken.ico (exe, installer) and duckstoken/web/icon.png
    python assets/make_icons.py --out DIR  writes duckstoken.ico and icon.png into DIR instead
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from duckstoken.shell import write_icons  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate the Duckstoken icons.")
    ap.add_argument("--out", help="write both files into this folder instead of updating the repository")
    out = ap.parse_args().out
    if out:
        written = write_icons(out)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            ico, png = write_icons(tmp)
            written = [Path(shutil.copyfile(ico, ROOT / "assets" / "duckstoken.ico")),
                       Path(shutil.copyfile(png, ROOT / "duckstoken" / "web" / "icon.png"))]
    for path in written:
        print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
