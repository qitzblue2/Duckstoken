"""Choosing the llama.cpp release to bundle/install (CI failure: `releases/latest` had no usable Windows zips).

llama.cpp creates each release as a pre-release and uploads its archives one by one afterwards, so the newest release is
often empty or incomplete. The names below follow its release workflow (`llama-<tag>-bin-win-<backend>-x64.zip`,
`cudart-llama-bin-win-cuda-<ver>-x64.zip`).
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import fetch_engine  # noqa: E402
from duckstoken import downloads  # noqa: E402


def asset(name, state="uploaded"):
    return {"name": name, "browser_download_url": "https://github.com/x/" + name, "size": 5, "state": state,
            "digest": "sha256:ab"}


def release(tag, names=(), draft=False, prerelease=True, state="uploaded"):
    return {"tag_name": tag, "draft": draft, "prerelease": prerelease, "assets": [asset(n, state) for n in names]}


def full(tag):
    return [f"llama-{tag}-bin-win-cpu-x64.zip", f"llama-{tag}-bin-win-cpu-arm64.zip",
            f"llama-{tag}-bin-win-vulkan-x64.zip", f"llama-{tag}-bin-win-cuda-12.4-x64.zip",
            "cudart-llama-bin-win-cuda-12.4-x64.zip", f"llama-{tag}-bin-macos-arm64.tar.gz",
            f"llama-{tag}-bin-win-openvino-2026.1-x64.zip", f"llama-{tag}-bin-win-sycl-x64.zip"]


class ChooseRelease(unittest.TestCase):
    def test_complete_prerelease_is_used(self):
        rel, win, picked = downloads.choose_llama_release([release("b2", full("b2"))], "cpu")
        self.assertEqual((rel["tag_name"], [a["name"] for a in picked]), ("b2", ["llama-b2-bin-win-cpu-x64.zip"]))
        self.assertNotIn("llama-b2-bin-win-cpu-arm64.zip", [a["name"] for a in win])

    def test_release_still_being_uploaded_is_skipped(self):
        rels = [release("b3"), release("b2", full("b2")), release("b1", full("b1"))]
        self.assertEqual(downloads.choose_llama_release(rels, "cpu")[0]["tag_name"], "b2")

    def test_assets_that_are_not_uploaded_yet_do_not_count(self):
        rels = [release("b3", full("b3"), state="starter"), release("b2", full("b2"))]
        self.assertEqual(downloads.choose_llama_release(rels, "cpu")[0]["tag_name"], "b2")

    def test_drafts_are_skipped(self):
        rels = [release("b3", full("b3"), draft=True), release("b2", full("b2"))]
        self.assertEqual(downloads.choose_llama_release(rels, "cpu")[0]["tag_name"], "b2")

    def test_vulkan_never_falls_back_to_the_cpu_zip_of_a_partial_release(self):
        rels = [release("b3", ["llama-b3-bin-win-cpu-x64.zip"]), release("b2", full("b2"))]
        self.assertEqual(downloads.choose_llama_release(rels, "cpu")[0]["tag_name"], "b3")
        rel, _win, picked = downloads.choose_llama_release(rels, "vulkan")
        self.assertEqual((rel["tag_name"], picked[0]["name"]), ("b2", "llama-b2-bin-win-vulkan-x64.zip"))

    def test_cuda_needs_its_runtime_zip(self):
        no_rt = [n for n in full("b3") if not n.startswith("cudart")]
        rels = [release("b3", no_rt), release("b2", full("b2"))]
        rel, _win, picked = downloads.choose_llama_release(rels, "cuda")
        self.assertEqual(rel["tag_name"], "b2")
        self.assertEqual([a["name"] for a in picked],
                         ["llama-b2-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"])

    def test_nothing_usable_says_which_releases_were_looked_at(self):
        with self.assertRaises(LookupError) as cm:
            downloads.choose_llama_release([release("b3"), release("b2", ["llama-b2-bin-macos-arm64.tar.gz"])], "cpu")
        self.assertIn("b3", str(cm.exception))
        self.assertIn("b2", str(cm.exception))

    def test_assets_without_state_field_still_work(self):
        rel = release("b2", full("b2"))
        for a in rel["assets"]:
            del a["state"]
        self.assertEqual(downloads.choose_llama_release([rel], "cpu")[0]["tag_name"], "b2")

    def test_build_script_picks_the_same_asset_and_exits_clearly_when_there_is_none(self):
        rel, a = fetch_engine.pick_asset([release("b3"), release("b2", full("b2"))], "cpu")
        self.assertEqual((rel["tag_name"], a["name"], a["digest"]), ("b2", "llama-b2-bin-win-cpu-x64.zip", "sha256:ab"))
        with self.assertRaises(SystemExit) as cm:
            fetch_engine.pick_asset([release("b3")], "cpu")
        self.assertIn("b3", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
