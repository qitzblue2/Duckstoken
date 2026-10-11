"""Write duckstoken/_build.py so the running app knows which build it is (used by the updater)."""

import argparse
import datetime
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--sha", required=True)
ap.add_argument("--installer", choices=["0", "1"], required=True)
a = ap.parse_args()
Path(__file__).resolve().parent.parent.joinpath("duckstoken", "_build.py").write_text(
    '"""Written by tools/stamp_build.py during release builds; the committed defaults describe a source checkout."""\n\n'
    f'BUILD_SHA = "{a.sha}"\n'
    f'BUILD_TIME = "{datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}"\n'
    f'INSTALLER = {a.installer == "1"}\n', encoding="utf-8")
print("stamped", a.sha[:12], "installer" if a.installer == "1" else "portable")
