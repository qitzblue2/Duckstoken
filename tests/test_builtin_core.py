"""Built-in engine core: GGUF reading and dequantization, tokenizers, chat templates, the transformer and the sampler."""
from __future__ import annotations

import dataclasses
import datetime
import math
import os
import random
import struct
import sys
import tempfile
import time
import types
import unicodedata
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("DUCKSTOKEN_HOME", tempfile.mkdtemp(prefix="duck-builtin-"))

try:
    import numpy as np
except ImportError:  # pragma: no cover - the whole module is skipped below
    np = None

try:
    import jinja2
    import jinja2.sandbox
except ImportError:
    jinja2 = None

if np is not None:
    import tiny_gguf  # noqa: E402
    from duckstoken.engines.builtin import gguf_weights as gw  # noqa: E402
    from duckstoken.engines.builtin.gguf_weights import GGUFFile, UnsupportedModel  # noqa: E402

needs_numpy = unittest.skipIf(np is None, "NumPy is not installed")
TMP = tempfile.TemporaryDirectory(prefix="duck-core-", ignore_cleanup_errors=True)
OPEN_MODELS: list = []


def tmp_path(name: str) -> Path:
    return Path(TMP.name) / name


def load_model(path, **kw):
    """LoadedModel.load, remembering the model so that its file is released before the temporary directory is deleted."""
    model = LoadedModel.load(path, **kw)
    OPEN_MODELS.append(model)
    return model


def tearDownModule():
    for model in OPEN_MODELS:
        model.close()
    TMP.cleanup()


# ---------------------------------------------------------------------------------------------------------------------
# Reference dequantizers: literal ports of ggml's C loops, one block at a time, in plain Python.
def f16(b, at):
    return struct.unpack_from("<e", b, at)[0]


def ref_q4_0(b):
    d, y = f16(b, 0), [0.0] * 32
    for j in range(16):
        y[j], y[j + 16] = ((b[2 + j] & 15) - 8) * d, ((b[2 + j] >> 4) - 8) * d
    return y


def ref_q4_1(b):
    d, m, y = f16(b, 0), f16(b, 2), [0.0] * 32
    for j in range(16):
        y[j], y[j + 16] = (b[4 + j] & 15) * d + m, (b[4 + j] >> 4) * d + m
    return y


def ref_q5_0(b):
    d, qh, y = f16(b, 0), struct.unpack_from("<I", b, 2)[0], [0.0] * 32
    for j in range(16):
        xh0, xh1 = ((qh >> j) << 4) & 0x10, (qh >> (j + 12)) & 0x10
        y[j] = (((b[6 + j] & 15) | xh0) - 16) * d
        y[j + 16] = (((b[6 + j] >> 4) | xh1) - 16) * d
    return y


def ref_q5_1(b):
    d, m, qh, y = f16(b, 0), f16(b, 2), struct.unpack_from("<I", b, 4)[0], [0.0] * 32
    for j in range(16):
        xh0, xh1 = ((qh >> j) << 4) & 0x10, (qh >> (j + 12)) & 0x10
        y[j] = ((b[8 + j] & 15) | xh0) * d + m
        y[j + 16] = ((b[8 + j] >> 4) | xh1) * d + m
    return y


def ref_q8_0(b):
    d = f16(b, 0)
    return [struct.unpack_from("b", b, 2 + i)[0] * d for i in range(32)]


def ref_scale_min_k4(j, q):
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 15) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


def ref_q4_k(b):
    d, dmin, scales, qs, y, is_, off = f16(b, 0), f16(b, 2), b[4:16], b[16:], [], 0, 0
    for _ in range(4):
        sc, m = ref_scale_min_k4(is_, scales)
        d1, m1 = d * sc, dmin * m
        sc, m = ref_scale_min_k4(is_ + 1, scales)
        d2, m2 = d * sc, dmin * m
        y += [d1 * (qs[off + l] & 15) - m1 for l in range(32)]
        y += [d2 * (qs[off + l] >> 4) - m2 for l in range(32)]
        off, is_ = off + 32, is_ + 2
    return y


def ref_q5_k(b):
    d, dmin, scales, qh, ql, y = f16(b, 0), f16(b, 2), b[4:16], b[16:48], b[48:], []
    is_, off, u1, u2 = 0, 0, 1, 2
    for _ in range(4):
        sc, m = ref_scale_min_k4(is_, scales)
        d1, m1 = d * sc, dmin * m
        sc, m = ref_scale_min_k4(is_ + 1, scales)
        d2, m2 = d * sc, dmin * m
        y += [d1 * ((ql[off + l] & 15) + (16 if qh[l] & u1 else 0)) - m1 for l in range(32)]
        y += [d2 * ((ql[off + l] >> 4) + (16 if qh[l] & u2 else 0)) - m2 for l in range(32)]
        off, is_, u1, u2 = off + 32, is_ + 2, u1 << 2, u2 << 2
    return y


def ref_q6_k(b):
    ql, qh, d, y = b[:128], b[128:192], f16(b, 208), [0.0] * 256
    sc = [struct.unpack_from("b", b, 192 + i)[0] for i in range(16)]
    for n in range(2):
        for l in range(32):
            i = l // 16
            q1 = ((ql[64 * n + l] & 15) | (((qh[32 * n + l] >> 0) & 3) << 4)) - 32
            q2 = ((ql[64 * n + l + 32] & 15) | (((qh[32 * n + l] >> 2) & 3) << 4)) - 32
            q3 = ((ql[64 * n + l] >> 4) | (((qh[32 * n + l] >> 4) & 3) << 4)) - 32
            q4 = ((ql[64 * n + l + 32] >> 4) | (((qh[32 * n + l] >> 6) & 3) << 4)) - 32
            s = 8 * n
            y[128 * n + l] = d * sc[s + i] * q1
            y[128 * n + l + 32] = d * sc[s + i + 2] * q2
            y[128 * n + l + 64] = d * sc[s + i + 4] * q3
            y[128 * n + l + 96] = d * sc[s + i + 6] * q4
    return y


def ref_q2_k(b):
    scales, qs, d, dmin, y, is_ = b[:16], b[16:80], f16(b, 80), f16(b, 82), [], 0
    for n in range(2):
        for j in range(4):
            for half in range(2):
                sc = scales[is_]
                is_ += 1
                dl, ml = d * (sc & 15), dmin * (sc >> 4)
                y += [dl * ((qs[32 * n + 16 * half + l] >> (2 * j)) & 3) - ml for l in range(16)]
    return y


def ref_q3_k(b):
    hm, qs, raw, d_all = b[:32], b[32:96], list(b[96:108]), f16(b, 108)
    aux = [struct.unpack_from("<I", bytes(raw), 4 * i)[0] for i in range(3)] + [0]
    k1, k2 = 0x03030303, 0x0F0F0F0F
    tmp = aux[2]
    aux[2] = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4)
    aux[3] = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4)
    aux[0] = (aux[0] & k2) | (((tmp >> 0) & k1) << 4)
    aux[1] = (aux[1] & k2) | (((tmp >> 2) & k1) << 4)
    scales = [x - 32 for x in struct.pack("<4I", *aux)]
    y, is_, m = [], 0, 1
    for n in range(2):
        shift = 0
        for j in range(4):
            for half in range(2):
                dl = d_all * scales[is_]
                is_ += 1
                y += [dl * (((qs[32 * n + 16 * half + l] >> shift) & 3) - (0 if hm[16 * half + l] & m else 4)) for l in range(16)]
            shift += 2
            m <<= 1
    return y


IQ4_NL = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]


def ref_iq4_nl(b):
    d, y = f16(b, 0), [0.0] * 32
    for j in range(16):
        y[j], y[j + 16] = d * IQ4_NL[b[2 + j] & 15], d * IQ4_NL[b[2 + j] >> 4]
    return y


def ref_iq4_xs(b):
    d, scales_h, scales_l, qs, y = f16(b, 0), struct.unpack_from("<H", b, 2)[0], b[4:8], b[8:], []
    for ib in range(8):
        ls = ((scales_l[ib // 2] >> (4 * (ib % 2))) & 15) | (((scales_h >> (2 * ib)) & 3) << 4)
        dl = d * (ls - 32)
        y += [dl * IQ4_NL[qs[16 * ib + j] & 15] for j in range(16)] + [dl * IQ4_NL[qs[16 * ib + j] >> 4] for j in range(16)]
    return y


REFERENCES = {"q4_0": ref_q4_0, "q4_1": ref_q4_1, "q5_0": ref_q5_0, "q5_1": ref_q5_1, "q8_0": ref_q8_0, "q4_k": ref_q4_k,
              "q5_k": ref_q5_k, "q6_k": ref_q6_k, "q2_k": ref_q2_k, "q3_k": ref_q3_k, "iq4_nl": ref_iq4_nl, "iq4_xs": ref_iq4_xs}
SCALE_FIELDS = {"q4_0": (0,), "q4_1": (0, 2), "q5_0": (0,), "q5_1": (0, 2), "q8_0": (0,), "q4_k": (0, 2), "q5_k": (0, 2),
                "q6_k": (208,), "q2_k": (80, 82), "q3_k": (108,), "iq4_nl": (0,), "iq4_xs": (0,)}


def random_blocks(qtype: str, n: int, seed: int = 1) -> np.ndarray:
    """n random blocks whose float16 scale fields hold ordinary finite numbers (everything else is random bits)."""
    rng = np.random.default_rng(seed)
    elements, size = tiny_gguf.BLOCK[qtype]
    blocks = rng.integers(0, 256, (n, size), dtype=np.uint8)
    for at in SCALE_FIELDS[qtype]:
        blocks[:, at:at + 2] = rng.uniform(-2, 2, n).astype("<f2").view(np.uint8).reshape(n, 2)
    return blocks


def write_one_tensor(path, array, qtype):
    raw = array.astype("<f4").tobytes() if qtype == "f32" else tiny_gguf.quantize(array, qtype)
    tiny_gguf.write_gguf(path, {"general.architecture": "test"}, [("w", array.shape, tiny_gguf.TYPE_ID[qtype], raw)])


# Relative RMS error a round trip through each format may cost on Gaussian data (measured, with headroom).
ROUND_TRIP_TOLERANCE = {"f32": 0, "f16": 1e-3, "bf16": 1e-2, "q8_0": 0.01, "q6_k": 0.03, "q5_0": 0.06, "q5_1": 0.05, "q5_k": 0.06,
                        "q4_0": 0.12, "q4_1": 0.10, "q4_k": 0.10, "iq4_nl": 0.10, "iq4_xs": 0.10, "q3_k": 0.20, "q2_k": 0.55}


@needs_numpy
class DequantizationTests(unittest.TestCase):
    def test_matches_the_reference_loops_on_random_blocks(self):
        for qtype, ref in REFERENCES.items():
            with self.subTest(qtype):
                blocks = random_blocks(qtype, 24)
                got = np.empty((len(blocks), tiny_gguf.BLOCK[qtype][0]), np.float32)
                gw._TYPES[tiny_gguf.TYPE_ID[qtype]].dequantize(blocks, got)
                want = np.array([ref(bytes(row)) for row in blocks], np.float32)
                self.assertEqual(got.shape, want.shape)
                self.assertEqual(got.dtype, np.float32)
                np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)

    def test_round_trip_through_the_reference_quantizers(self):
        x = np.random.default_rng(3).standard_normal((4, 512)).astype(np.float32)
        for qtype, tol in ROUND_TRIP_TOLERANCE.items():
            with self.subTest(qtype):
                path = tmp_path(f"rt-{qtype}.gguf")
                write_one_tensor(path, x, qtype)
                with GGUFFile(path) as f:
                    got = f.load("w")
                self.assertEqual(got.shape, x.shape)
                err = np.sqrt(((got - x) ** 2).mean() / (x ** 2).mean())
                self.assertLessEqual(err, tol, f"{qtype}: relative error {err:.4f}")
                if qtype == "f32":
                    np.testing.assert_array_equal(got, x)

    def test_formats_are_ordered_by_precision(self):
        x = np.random.default_rng(4).standard_normal((2, 1024)).astype(np.float32)
        errors = {}
        for qtype in ("q8_0", "q6_k", "q5_k", "q4_k", "q3_k", "q2_k"):
            path = tmp_path(f"order-{qtype}.gguf")
            write_one_tensor(path, x, qtype)
            with GGUFFile(path) as f:
                errors[qtype] = float(np.abs(f.load("w") - x).mean())
        self.assertEqual(sorted(errors, key=errors.get), ["q8_0", "q6_k", "q5_k", "q4_k", "q3_k", "q2_k"])

    def test_load_to_bfloat16_and_into_a_buffer(self):
        x = np.random.default_rng(5).standard_normal((3, 64)).astype(np.float32)
        path = tmp_path("bf16.gguf")
        write_one_tensor(path, x, "q8_0")
        with GGUFFile(path) as f:
            full = f.load("w")
            half = f.load("w", "bfloat16")
            buf = np.zeros((3, 64), np.float32)
            self.assertIs(f.load("w", out=buf), buf)
            self.assertEqual(half.dtype, np.uint16)
            widened = (half.astype(np.uint32) << 16).view(np.float32)
            np.testing.assert_allclose(widened, full, rtol=2 ** -8)  # bfloat16 keeps 8 significant bits
            np.testing.assert_array_equal(buf, full)
            with self.assertRaises(ValueError):
                f.load("w", out=np.zeros((3, 63), np.float32))
            with self.assertRaises(ValueError):
                f.load("w", out=np.zeros((3, 64), np.uint16))
            with self.assertRaises(ValueError):
                f.load("w", "float16")

    def test_load_rows_reads_only_the_requested_rows(self):
        x = np.random.default_rng(7).standard_normal((20, 256)).astype(np.float32)
        for qtype in ("f32", "q8_0", "q4_k", "q6_k"):
            path = tmp_path(f"rows-{qtype}.gguf")
            write_one_tensor(path, x, qtype)
            with GGUFFile(path) as f:
                whole = f.load("w")
                np.testing.assert_array_equal(f.load_rows("w", [3, 0, 3, 19]), whole[[3, 0, 3, 19]])
                with self.assertRaises(IndexError):
                    f.load_rows("w", [20])

    def test_identical_tensors_are_detected(self):
        a = np.random.default_rng(8).standard_normal((4, 64)).astype(np.float32)
        raw = tiny_gguf.quantize(a, "q8_0")
        other = tiny_gguf.quantize(a * 2, "q8_0")
        tiny_gguf.write_gguf(tmp_path("same.gguf"), {"k": 1}, [("a", a.shape, 8, raw), ("b", a.shape, 8, raw), ("c", a.shape, 8, other),
                                                              ("d", a.shape, 0, a.tobytes())])
        with GGUFFile(tmp_path("same.gguf")) as f:
            self.assertTrue(f.same_data("a", "b"))
            self.assertFalse(f.same_data("a", "c"))
            self.assertFalse(f.same_data("a", "d"))
            self.assertFalse(f.same_data("a", "missing"))

    def test_large_tensors_are_dequantized_in_chunks(self):
        x = np.random.default_rng(6).standard_normal((300, 256)).astype(np.float32)
        path = tmp_path("chunks.gguf")
        write_one_tensor(path, x, "q4_k")
        with GGUFFile(path) as f:
            whole = f.load("w")
            old = gw._CHUNK_ELEMENTS
            gw._CHUNK_ELEMENTS = 256 * 7
            try:
                chunked = f.load("w")
            finally:
                gw._CHUNK_ELEMENTS = old
        np.testing.assert_array_equal(whole, chunked)


@needs_numpy
class GGUFParsingTests(unittest.TestCase):
    def setUp(self):
        self.path = tmp_path("parse.gguf")
        self.arr = np.arange(2 * 3 * 32, dtype=np.float32).reshape(2, 3, 32)
        meta = {"general.architecture": "test", "an.int": 7, "a.negative": -5, "a.big": 1 << 40, "a.float": 1.5, "a.bool": True,
                "a.string": "héllo ☃", "a.ints": [1, 2, 3], "a.floats": [0.5, 1.5], "a.strings": ["a", "bc", ""],
                "general.alignment": 64}
        tiny_gguf.write_gguf(self.path, meta, [("t", self.arr.shape, 0, self.arr.tobytes()),
                                              ("v", (4,), 0, np.arange(4, dtype=np.float32).tobytes())], alignment=64)

    def test_metadata_tensors_and_shapes(self):
        with GGUFFile(self.path) as f:
            m = f.meta
            self.assertEqual((m["an.int"], m["a.negative"], m["a.big"], m["a.float"], m["a.bool"]), (7, -5, 1 << 40, 1.5, True))
            self.assertEqual((m["a.string"], m["a.ints"], m["a.floats"], m["a.strings"]), ("héllo ☃", [1, 2, 3], [0.5, 1.5], ["a", "bc", ""]))
            info = f.tensors["t"]
            self.assertEqual((info.shape, info.ggml_type, info.n_bytes, info.type_name), ((2, 3, 32), 0, 2 * 3 * 32 * 4, "F32"))
            self.assertEqual(info.offset % 64, 0)
            self.assertEqual(f.tensors["v"].shape, (4,))
            np.testing.assert_array_equal(f.load("t"), self.arr)
            np.testing.assert_array_equal(f.load("v"), np.arange(4))

    def test_close_releases_the_file(self):
        f = GGUFFile(self.path)
        f.load("t")
        f.close()
        f.close()
        self.assertTrue(f.closed)
        with self.assertRaises(ValueError):
            f.load("t")
        os.remove(self.path)  # fails on Windows if the mapping were still open
        self.assertFalse(self.path.exists())

    def test_rejects_unreadable_files_with_a_plain_message(self):
        good = self.path.read_bytes()
        cases = {"missing": None, "empty": b"", "not gguf": b"MZ" + b"\0" * 100, "future version": good[:4] + struct.pack("<I", 9) + good[8:],
                 "big endian": b"GGUF" + struct.pack(">I", 3) + good[8:], "huge counts": good[:8] + struct.pack("<QQ", 1 << 50, 1) + good[24:]}
        for label, data in cases.items():
            with self.subTest(label):
                p = tmp_path("bad.gguf")
                if data is not None:
                    p.write_bytes(data)
                elif p.exists():
                    p.unlink()
                with self.assertRaises(UnsupportedModel) as cm:
                    GGUFFile(p)
                self.assertTrue(str(cm.exception))

    def test_every_truncation_is_detected(self):
        good = self.path.read_bytes()
        with GGUFFile(self.path) as f:
            data_end = max(t.offset + t.n_bytes for t in f.tensors.values())
        cut = tmp_path("cut.gguf")
        for n in list(range(0, 300, 3)) + [data_end - 100, data_end - 1]:
            cut.write_bytes(good[:n])
            with self.assertRaises(UnsupportedModel, msg=f"cut at {n}"):
                GGUFFile(cut)
        cut.write_bytes(good[:data_end])  # the alignment padding after the last tensor is optional
        with GGUFFile(cut) as f:
            np.testing.assert_array_equal(f.load("v"), np.arange(4))

    def test_lying_array_and_string_sizes_do_not_allocate(self):
        head = b"GGUF" + struct.pack("<IQQ", 3, 0, 1)
        evil_array = head + tiny_gguf._s("k") + struct.pack("<II", 9, 4) + struct.pack("<Q", 1 << 60)
        evil_strings = head + tiny_gguf._s("k") + struct.pack("<II", 9, 8) + struct.pack("<Q", 1 << 40)
        evil_string = head + tiny_gguf._s("k") + struct.pack("<I", 8) + struct.pack("<Q", 1 << 62)
        nested = head + tiny_gguf._s("k") + struct.pack("<I", 9) + (struct.pack("<IQ", 9, 1)) * 5 + b"\0" * 64
        for label, data in (("array", evil_array), ("strings", evil_strings), ("string", evil_string), ("nested", nested)):
            with self.subTest(label):
                p = tmp_path("evil.gguf")
                p.write_bytes(data)
                with self.assertRaises(UnsupportedModel):
                    GGUFFile(p)

    def test_tensor_outside_the_file_or_with_bad_shape(self):
        for label, shape, raw in (("beyond eof", (2, 64), b"\0" * 16), ("zero dim", (0, 32), b"")):
            with self.subTest(label):
                p = tmp_path("tensor.gguf")
                tiny_gguf.write_gguf(p, {"a": 1}, [("w", shape, 0, raw)])
                with self.assertRaises(UnsupportedModel):
                    GGUFFile(p)

    def test_row_length_must_fit_the_block_size(self):
        p = tmp_path("rows.gguf")
        tiny_gguf.write_gguf(p, {"a": 1}, [("w", (2, 40), tiny_gguf.TYPE_ID["q8_0"], b"\0" * 200)])
        with self.assertRaisesRegex(UnsupportedModel, "multiple of 32"):
            GGUFFile(p)

    def test_unsupported_and_unknown_types_are_named(self):
        p = tmp_path("types.gguf")
        tiny_gguf.write_gguf(p, {"a": 1}, [("iq2", (1, 256), 16, b"\0" * 66), ("odd", (1, 8), 99, b"\0" * 8)])
        with GGUFFile(p) as f:
            self.assertEqual(f.tensors["iq2"].type_name, "IQ2_XXS")
            self.assertEqual(f.tensors["odd"].type_name, "type 99")
            with self.assertRaisesRegex(UnsupportedModel, "IQ2_XXS"):
                f.load("iq2")
            with self.assertRaisesRegex(UnsupportedModel, "type 99"):
                f.load("odd")
            with self.assertRaisesRegex(UnsupportedModel, "no tensor named"):
                f.load("nope")
        self.assertFalse(gw.is_supported(16))
        self.assertTrue(all(gw.is_supported(tiny_gguf.TYPE_ID[t]) for t in ("q4_k", "q6_k", "q8_0", "f16", "bf16")))

    def test_duplicate_keys_and_split_models_are_rejected(self):
        p = tmp_path("dup.gguf")
        data = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 0, 2))
        for _ in range(2):
            data += tiny_gguf._s("k") + struct.pack("<II", 4, 1)
        p.write_bytes(bytes(data))
        with self.assertRaisesRegex(UnsupportedModel, "duplicate"):
            GGUFFile(p)
        tiny_gguf.write_gguf(p, {"split.count": 2}, [])
        with self.assertRaisesRegex(UnsupportedModel, "[Ss]plit"):
            GGUFFile(p)

    def test_version_2_files_are_read(self):
        p = tmp_path("v2.gguf")
        tiny_gguf.write_gguf(p, {"a": 1}, [("w", (2,), 0, np.ones(2, np.float32).tobytes())], version=2)
        with GGUFFile(p) as f:
            np.testing.assert_array_equal(f.load("w"), [1, 1])


# ---------------------------------------------------------------------------------------------------------------------
# Tokenizers
if np is not None:
    from duckstoken.engines.builtin import tokenizer as tk  # noqa: E402
    from duckstoken.engines.builtin.tokenizer import Tokenizer  # noqa: E402

    TODAY = tk._today  # the unpatched function

WHITE_SPACE = set(map(ord, "\t\n\x0b\x0c\r \x85\xa0     　")) | set(range(0x2000, 0x200B))
NASTY_ALPHABET = list("aZxé中жاßǅʰ") + list("07٣²Ⅷ½") + ["́", "ा"] + list("!-_<>/€'\"") + ["😀", "‍", "\x1c", "\x00"] \
    + [" ", " ", "\t", "\n", "\r", " ", " ", "　", "\x85"] + ["'s", "'T", "'re", "'LL", "'d"]


def is_letter(c: str) -> bool:
    return unicodedata.category(c).startswith("L")


def is_number(c: str) -> bool:
    return unicodedata.category(c).startswith("N")


def is_space(c: str) -> bool:
    return ord(c) in WHITE_SPACE


def contraction_length(text: str, i: int) -> int:
    """Length of a case-insensitive 's 't 're 've 'm 'll 'd at i, else 0."""
    if text[i] != "'":
        return 0
    for suffix in ("s", "t", "re", "ve", "m", "ll", "d"):
        if text[i + 1:i + 1 + len(suffix)].lower() == suffix:
            return 1 + len(suffix)
    return 0


def space_run_end(text: str, i: int) -> int:
    while i < len(text) and is_space(text[i]):
        i += 1
    return i


def reference_split(text: str, flavour: str) -> list[str]:
    """The pre-tokenizer patterns written out by hand (no regular expressions): flavour 'qwen2', 'llama3' or 'gpt2'."""
    out, i = [], 0
    while i < len(text):
        k = lowercase_contraction(text, i) if flavour == "gpt2" else contraction_length(text, i)
        if k:
            end = i + k
        elif flavour == "gpt2":
            end = gpt2_alternatives(text, i)
        else:
            end = qwen_alternatives(text, i, 1 if flavour == "qwen2" else 3)
        out.append(text[i:end])
        i = end
    return out


def lowercase_contraction(text: str, i: int) -> int:
    for suffix in ("s", "t", "re", "ve", "m", "ll", "d"):
        if text.startswith("'" + suffix, i):
            return 1 + len(suffix)
    return 0


def run_of(text: str, i: int, test) -> int:
    while i < len(text) and test(text[i]):
        i += 1
    return i


def whitespace_tail(text: str, i: int) -> int:
    """`\\s+(?!\\S)` then `\\s+` at i (text[i] is white space)."""
    e = space_run_end(text, i)
    if e == len(text):
        return e
    return e - 1 if e - i >= 2 else e


def gpt2_alternatives(text: str, i: int) -> int:
    n = len(text)
    j = i + 1 if text[i] == " " and i + 1 < n else i
    if is_letter(text[j]):
        return run_of(text, j, is_letter)
    if is_number(text[j]):
        return run_of(text, j, is_number)
    other = lambda ch: not (is_space(ch) or is_letter(ch) or is_number(ch))  # noqa: E731
    if other(text[j]):
        return run_of(text, j, other)
    return whitespace_tail(text, i)


def qwen_alternatives(text: str, i: int, max_digits: int) -> int:
    n, c = len(text), text[i]
    # [^\r\n\p{L}\p{N}]?\p{L}+
    if is_letter(c):
        return run_of(text, i, is_letter)
    if c not in "\r\n" and not is_number(c) and i + 1 < n and is_letter(text[i + 1]):
        return run_of(text, i + 1, is_letter)
    if is_number(c):  # \p{N} or \p{N}{1,3}
        j = i
        while j < n and j - i < max_digits and is_number(text[j]):
            j += 1
        return j
    other = lambda ch: not (is_space(ch) or is_letter(ch) or is_number(ch))  # noqa: E731
    j = i + 1 if c == " " and i + 1 < n else i
    if other(text[j]):  # ` ?[^\s\p{L}\p{N}]+[\r\n]*`
        return run_of(text, run_of(text, j, other), lambda ch: ch in "\r\n")
    e = space_run_end(text, i)  # c is white space here
    newline = max((p for p in range(i, e) if text[p] in "\r\n"), default=None)
    if newline is not None:  # \s*[\r\n]+
        return newline + 1
    return whitespace_tail(text, i)


def random_text(rng, length: int) -> str:
    return "".join(rng.choice(NASTY_ALPHABET) for _ in range(length))


TRICKY_TEXTS = [
    "", "a", "Hello, world!", "  leading and trailing  ", "tabs\tand\nnewlines\r\n\r\n", "  \n  \n x",
    "unicode: café naïve Ünïcödé 你好世界 こんにちは مرحبا", "emoji 😀👍🏽 family 👨‍👩‍👧‍👦 flag 🇫🇷",
    "numbers 0123456789 3.14159 1,000,000 ٣٤٥ ²½", "code: def f(x):\n    return x ** 2  # comment\n\n\n",
    "special looking <|im_start|> <|im_end|> <think>x</think> </s> <s> <0x41>", "NUL \x00 and control \x1b[0m",
    "'s 're 'LL contractions don't won't I'M", "mixed nbsp　ideographic em space", "a" * 300, "\n\n\n", " ", "   ",
    "trailing newline\n", "� replacement and ​ zero width", "Ünï ́combining", "ß" * 5 + "ǅ",
]


@needs_numpy
class PreTokenizerTests(unittest.TestCase):
    def test_splits_like_the_hand_written_reference(self):
        rng = random.Random(11)
        samples = TRICKY_TEXTS + [random_text(rng, rng.randint(1, 40)) for _ in range(1500)]
        for name, flavour in (("qwen2", "qwen2"), ("llama-bpe", "llama3"), ("default", "gpt2")):
            regexes = tk._compiled(tuple(tk.PRE_TOKENIZERS[name]))
            for text in samples:
                with self.subTest(name, text=text):
                    got = tk._split(text, regexes)
                    self.assertEqual("".join(got), text)
                    self.assertEqual(got, reference_split(text, flavour))

    def test_known_splits(self):
        qwen = tk._compiled(tuple(tk.PRE_TOKENIZERS["qwen2"]))
        self.assertEqual(tk._split("Hello, world! It's 12345 foo\n\n  bar  \t baz  ", qwen),
                         ["Hello", ",", " world", "!", " It", "'s", " ", "1", "2", "3", "4", "5", " foo", "\n\n", " ", " bar",
                          "  \t", " baz", "  "])
        llama = tk._compiled(tuple(tk.PRE_TOKENIZERS["llama-bpe"]))
        self.assertEqual(tk._split("x 1234567 y", llama), ["x", " ", "123", "456", "7", " y"])
        smol = tk._compiled(tuple(tk.PRE_TOKENIZERS["smollm"]))
        self.assertEqual(tk._split("a 12 b", smol), ["a", " ", "1", "2", " b"])
        gpt2 = tk._compiled(tuple(tk.PRE_TOKENIZERS["gpt-2"]))
        self.assertEqual(tk._split("a  12 b'sx", gpt2), ["a", " ", " 12", " b", "'s", "x"])

    def test_unicode_classes_match_the_database(self):
        letters, numbers = tk._unicode_classes()
        import re
        is_l, is_n = re.compile(f"[{letters}]"), re.compile(f"[{numbers}]")
        for cp in list(range(0x3000)) + list(range(0x1D400, 0x1D800)) + list(range(0x20000, 0x20100)):
            c = chr(cp)
            self.assertEqual(bool(is_l.fullmatch(c)), unicodedata.category(c).startswith("L"), hex(cp))
            self.assertEqual(bool(is_n.fullmatch(c)), unicodedata.category(c).startswith("N"), hex(cp))


def bpe(**kw) -> "Tokenizer":
    return Tokenizer.from_gguf(tiny_gguf.bpe_tokenizer_meta(**kw))


def spm(**kw) -> "Tokenizer":
    return Tokenizer.from_gguf(tiny_gguf.spm_tokenizer_meta(**kw))


def naive_bpe(word: str, ranks: dict) -> list[str]:
    syms = list(word)
    while True:
        pairs = [(ranks[(a, b)], i) for i, (a, b) in enumerate(zip(syms, syms[1:])) if (a, b) in ranks]
        if not pairs:
            return syms
        _, i = min(pairs)
        syms[i:i + 2] = [syms[i] + syms[i + 1]]


@needs_numpy
class ByteLevelBPETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = bpe()

    def test_round_trips_tricky_text_exactly(self):
        for text in TRICKY_TEXTS:
            with self.subTest(text=text):
                ids = self.tok.encode(text, parse_special=False)
                self.assertTrue(all(0 <= i < self.tok.vocab_size for i in ids))
                self.assertEqual(self.tok.decode(ids), text)
                self.assertEqual(self.tok.decode(ids, special=True), text)

    def test_random_text_round_trips(self):
        rng = random.Random(5)
        for _ in range(300):
            text = random_text(rng, rng.randint(0, 60))
            self.assertEqual(self.tok.decode(self.tok.encode(text, parse_special=False)), text)

    def test_merges_apply_and_shorten_the_sequence(self):
        plain = "the quick brown fox jumps over the lazy dog"
        ids = self.tok.encode(plain)
        self.assertLess(len(ids), len(plain.encode()))
        self.assertGreater(len(ids), 0)

    def test_merge_order_matches_the_naive_algorithm(self):
        tok, rng = self.tok, random.Random(3)
        table, _ = tk._byte_unicode()
        corpus = " ".join(tiny_gguf.CORPUS)
        for _ in range(400):
            start = rng.randrange(len(corpus))
            raw = corpus[start:start + rng.randint(1, 30)].encode()
            word = "".join(table[b] for b in raw)
            expected = [tok._token_ids[s] for s in naive_bpe(word, tok._ranks)]
            self.assertEqual(tok._merge(word), expected, word)

    def test_a_very_long_word_is_tokenized_quickly(self):
        word = "the" * 70_000 + "a" * 100_000  # plenty of merges to apply, with no space to split on
        start = time.perf_counter()
        ids = self.tok.encode(word, parse_special=False)
        self.assertLess(time.perf_counter() - start, 5.0)
        self.assertEqual(self.tok.decode(ids), word)

    def test_special_tokens(self):
        t = self.tok
        start, end, end_of_text = t._token_ids["<|im_start|>"], t._token_ids["<|im_end|>"], t._token_ids["<|endoftext|>"]
        ids = t.encode("<|im_start|>user\nhi<|im_end|>")
        self.assertEqual((ids[0], ids[-1]), (start, end))
        self.assertEqual(t.decode(ids), "user\nhi")
        self.assertEqual(t.decode(ids, special=True), "<|im_start|>user\nhi<|im_end|>")
        literal = t.encode("<|im_start|>", parse_special=False)
        self.assertNotIn(start, literal)
        self.assertEqual(t.decode(literal), "<|im_start|>")
        think = t._token_ids["<think>"]  # user-defined: always one token, and visible in the output
        self.assertEqual(t.encode("<think>", parse_special=False), [think])
        self.assertEqual(t.decode([think]), "<think>")
        self.assertEqual(t.eos_ids, {end, end_of_text})
        self.assertEqual(t.eos_id, end)
        self.assertEqual(t.vocab_size, len(t.tokens))

    def test_bos_is_optional_and_never_doubled(self):
        t = self.tok
        self.assertFalse(t.add_bos)
        bos = t.bos_id
        self.assertEqual(t.encode("hi", add_bos=True)[0], bos)
        self.assertNotEqual(t.encode("hi")[0], bos)
        self.assertEqual(t.encode("<|endoftext|>hi", add_bos=True).count(bos), 1)  # this vocabulary's BOS is <|endoftext|>
        self.assertEqual(t.encode("", add_bos=True), [bos])
        self.assertEqual(t.encode(""), [])

    def test_lone_surrogates_do_not_crash(self):
        self.assertEqual(self.tok.decode(self.tok.encode("a\ud800b", parse_special=False)), "a?b")

    def test_invalid_ids_are_rejected(self):
        for bad in (-1, self.tok.vocab_size):
            with self.assertRaises(ValueError):
                self.tok.decode([bad])

    def test_unknown_pre_tokenizer_and_missing_data_are_refused(self):
        meta = tiny_gguf.bpe_tokenizer_meta()
        for change in ({"tokenizer.ggml.pre": "something-new"}, {"tokenizer.ggml.merges": []}, {"tokenizer.ggml.model": "bert"},
                       {"tokenizer.ggml.tokens": []}, {"tokenizer.ggml.token_type": [1, 2]}):
            with self.subTest(change):
                with self.assertRaises(UnsupportedModel):
                    Tokenizer.from_gguf({**meta, **change})

    def test_each_pre_tokenizer_name_builds_a_tokenizer(self):
        for pre in tk.PRE_TOKENIZERS:
            t = bpe(pre=pre)
            self.assertEqual(t.decode(t.encode("Hello 123 world!", parse_special=False)), "Hello 123 world!")

    def test_pathological_whitespace_is_linear(self):
        for text in (" " * 400_000, "\t\n" * 200_000, " \n" * 200_000 + "x", " " * 100_000 + "é" + " " * 100_000):
            start = time.perf_counter()
            ids = self.tok.encode(text, parse_special=False)
            self.assertLess(time.perf_counter() - start, 5.0)
            self.assertEqual(self.tok.decode(ids), text)

    def test_stream_decoder_waits_for_complete_characters(self):
        t = self.tok
        text = "héllo 😀 wörld 你好"
        ids = t.encode(text)
        d = t.stream_decoder()
        pieces = [d.push(i) for i in ids]
        self.assertEqual("".join(pieces) + d.flush(), text)
        emoji = t.encode("😀", parse_special=False)
        self.assertGreater(len(emoji), 1)  # a 4-byte character spans several tokens in a byte-level vocabulary
        d = t.stream_decoder()
        shown = [d.push(i) for i in emoji]
        self.assertEqual(shown[:-1], [""] * (len(emoji) - 1))
        self.assertEqual(shown[-1], "😀")

    def test_stream_decoder_flush_marks_a_dangling_half_character(self):
        t = self.tok
        d = t.stream_decoder()
        first = t.encode("😀", parse_special=False)[0]
        self.assertEqual(d.push(first), "")
        self.assertEqual(d.flush(), "�")
        self.assertEqual(d.flush(), "")


def handmade_spm(pieces: dict, byte_tokens: bool = True, **extra) -> "Tokenizer":
    """A SentencePiece vocabulary of <unk>, <s>, </s>, 256 byte tokens, \u2581 and the given {piece: score}."""
    tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{b:02X}>" if byte_tokens else f"#{b}" for b in range(256)] + [tiny_gguf.SEPARATOR] + list(pieces)
    types = [2, 3, 3] + [6] * 256 + [1] * (1 + len(pieces))
    scores = [0.0] * 259 + [0.0] + list(pieces.values())
    return Tokenizer.from_gguf({"tokenizer.ggml.model": "llama", "tokenizer.ggml.tokens": tokens, "tokenizer.ggml.token_type": types,
                                "tokenizer.ggml.scores": scores, "tokenizer.ggml.bos_token_id": 1, "tokenizer.ggml.eos_token_id": 2,
                                "tokenizer.ggml.unknown_token_id": 0, **extra})


@needs_numpy
class SentencePieceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = spm()

    def test_round_trips_text_with_byte_fallback(self):
        for text in TRICKY_TEXTS:
            if "\u2581" in text:
                continue
            with self.subTest(text=text):
                ids = self.tok.encode(text, parse_special=False)
                self.assertEqual(self.tok.decode(ids), text)

    def test_random_text_round_trips(self):
        rng = random.Random(9)
        for _ in range(300):
            text = random_text(rng, rng.randint(0, 50))
            self.assertEqual(self.tok.decode(self.tok.encode(text, parse_special=False)), text)

    def test_the_first_word_gets_a_prefix_space(self):
        t = self.tok
        ids = t.encode("the end")
        self.assertEqual([t.tokens[i] for i in ids][0][:1], "\u2581")
        self.assertEqual(t.decode(ids), "the end")
        self.assertEqual(t.decode([t._token_ids["\u2581"]]), "")  # the dummy prefix alone vanishes
        self.assertEqual(t.encode(""), [])

    def test_highest_score_merges_first_and_ties_go_left(self):
        s = tiny_gguf.SEPARATOR
        pieces = {"a": 0.0, "b": 0.0, "c": 0.0, "ab": -2.0, "bc": -1.0, "abc": -3.0, s + "a": -5.0}
        t = handmade_spm(pieces)
        self.assertEqual([t.tokens[i] for i in t.encode("abc")], [s, "abc"])  # bc (-1) first, then a+bc
        pieces["ab"] = -0.5
        t = handmade_spm(pieces)
        self.assertEqual([t.tokens[i] for i in t.encode("abc")], [s, "abc"])  # ab (-0.5) first, then ab+c
        t = handmade_spm({"a": 0.0, "b": 0.0, "c": 0.0, "ab": -1.0, "bc": -1.0})
        self.assertEqual([t.tokens[i] for i in t.encode("abc")], [s, "ab", "c"])  # equal scores: the left pair merges first

    def test_unknown_characters_become_bytes_or_unk(self):
        t = self.tok
        ids = t.encode("é", parse_special=False)
        self.assertEqual([t.tokens[i] for i in ids][-2:], ["<0xC3>", "<0xA9>"])
        no_bytes = handmade_spm({"a": 0.0}, byte_tokens=False)
        self.assertEqual(no_bytes.encode("é", parse_special=False)[-2:], [0, 0])  # one <unk> per byte

    def test_special_tokens_and_spacing_around_them(self):
        t = self.tok
        ids = t.encode("<s>hi</s>there", add_bos=True)
        self.assertEqual((ids[0], t.tokens[ids[0]]), (1, "<s>"))
        self.assertEqual(ids.count(1), 1)
        self.assertIn(2, ids)
        text_after_eos = [t.tokens[i] for i in ids[ids.index(2) + 1:]]
        self.assertEqual(text_after_eos[0][:1], "\u2581")  # text after a special token starts a new "sentence"
        self.assertEqual(t.decode(ids), "hi there")
        self.assertEqual(t.decode(ids, special=True), "<s> hi</s> there")
        literal = t.encode("<s>", parse_special=False)
        self.assertNotIn(1, literal)
        self.assertEqual(t.decode(literal), "<s>")
        self.assertEqual(t.eos_ids, {2})

    def test_no_prefix_space_when_the_model_says_so(self):
        t = spm(add_space_prefix=False)
        ids = t.encode("the end")
        self.assertNotEqual(t.tokens[ids[0]][:1], "\u2581")
        self.assertEqual(t.decode(ids), "the end")
        self.assertEqual(t.decode(t.encode(" lead")), " lead")

    def test_leading_spaces_survive(self):
        for text in (" x", "  x", "\nx"):
            self.assertEqual(self.tok.decode(self.tok.encode(text, parse_special=False)), text)

    def test_stream_decoder_matches_decode_and_strips_the_prefix_space_once(self):
        t = self.tok
        text = "the end of 😀 days"
        ids = t.encode(text)
        d = t.stream_decoder()
        parts = [d.push(i) for i in ids]
        self.assertEqual("".join(parts) + d.flush(), text)
        self.assertEqual(t.decode(ids), text)

    def test_missing_scores_are_refused(self):
        with self.assertRaises(UnsupportedModel):
            Tokenizer.from_gguf({**tiny_gguf.spm_tokenizer_meta(), "tokenizer.ggml.scores": []})


# ---------------------------------------------------------------------------------------------------------------------
# Chat templates
# Real chat templates as stored in the GGUF files of these models (Jinja, never executed by the engine).
REAL_TEMPLATES = {
    'QWEN25': '{%- if tools %}\n    {{- \'<|im_start|>system\\n\' }}\n    {%- if messages[0][\'role\'] == \'system\' %}\n        {{- messages[0][\'content\'] }}\n    {%- else %}\n        {{- \'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\' }}\n    {%- endif %}\n    {{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}\n    {%- for tool in tools %}\n        {{- "\\n" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}}\\n</tool_call><|im_end|>\\n" }}\n{%- else %}\n    {%- if messages[0][\'role\'] == \'system\' %}\n        {{- \'<|im_start|>system\\n\' + messages[0][\'content\'] + \'<|im_end|>\\n\' }}\n    {%- else %}\n        {{- \'<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n\' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}\n        {{- \'<|im_start|>\' + message.role + \'\\n\' + message.content + \'<|im_end|>\' + \'\\n\' }}\n    {%- elif message.role == "assistant" %}\n        {{- \'<|im_start|>\' + message.role }}\n        {%- if message.content %}\n            {{- \'\\n\' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- \'\\n<tool_call>\\n{"name": "\' }}\n            {{- tool_call.name }}\n            {{- \'", "arguments": \' }}\n            {{- tool_call.arguments | tojson }}\n            {{- \'}\\n</tool_call>\' }}\n        {%- endfor %}\n        {{- \'<|im_end|>\\n\' }}\n    {%- elif message.role == "tool" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}\n            {{- \'<|im_start|>user\' }}\n        {%- endif %}\n        {{- \'\\n<tool_response>\\n\' }}\n        {{- message.content }}\n        {{- \'\\n</tool_response>\' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}\n            {{- \'<|im_end|>\\n\' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|im_start|>assistant\\n\' }}\n{%- endif %}\n',
    'QWEN3': '{%- if tools %}\n    {{- \'<|im_start|>system\\n\' }}\n    {%- if messages[0].role == \'system\' %}\n        {{- messages[0].content + \'\\n\\n\' }}\n    {%- endif %}\n    {{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}\n    {%- for tool in tools %}\n        {{- "\\n" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}\n{%- else %}\n    {%- if messages[0].role == \'system\' %}\n        {{- \'<|im_start|>system\\n\' + messages[0].content + \'<|im_end|>\\n\' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for index in range(ns.last_query_index, -1, -1) %}\n    {%- set message = messages[index] %}\n    {%- if ns.multi_step_tool and message.role == "user" and not(\'<tool_response>\' in message.content and \'</tool_response>\' in message.content) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}\n        {{- \'<|im_start|>\' + message.role + \'\\n\' + message.content + \'<|im_end|>\' + \'\\n\' }}\n    {%- elif message.role == "assistant" %}\n        {%- set content = message.content %}\n        {%- set reasoning_content = \'\' %}\n        {%- if message.reasoning_content is defined and message.reasoning_content is not none %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if \'</think>\' in message.content %}\n                {%- set content = message.content.split(\'</think>\')[-1].lstrip(\'\\n\') %}\n                {%- set reasoning_content = message.content.split(\'</think>\')[0].rstrip(\'\\n\').split(\'<think>\')[-1].lstrip(\'\\n\') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if loop.last or (not loop.last and reasoning_content) %}\n                {{- \'<|im_start|>\' + message.role + \'\\n<think>\\n\' + reasoning_content.strip(\'\\n\') + \'\\n</think>\\n\\n\' + content.lstrip(\'\\n\') }}\n            {%- else %}\n                {{- \'<|im_start|>\' + message.role + \'\\n\' + content }}\n            {%- endif %}\n        {%- else %}\n            {{- \'<|im_start|>\' + message.role + \'\\n\' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- \'\\n\' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- \'<tool_call>\\n{"name": "\' }}\n                {{- tool_call.name }}\n                {{- \'", "arguments": \' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- \'}\\n</tool_call>\' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- \'<|im_end|>\\n\' }}\n    {%- elif message.role == "tool" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}\n            {{- \'<|im_start|>user\' }}\n        {%- endif %}\n        {{- \'\\n<tool_response>\\n\' }}\n        {{- message.content }}\n        {{- \'\\n</tool_response>\' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}\n            {{- \'<|im_end|>\\n\' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|im_start|>assistant\\n\' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- \'<think>\\n\\n</think>\\n\\n\' }}\n    {%- endif %}\n{%- endif %}',
    'LLAMA32': '{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now("%d %b %Y") %}\n    {%- else %}\n        {%- set date_string = "26 Jul 2024" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{{- "Cutting Knowledge Date: December 2023\\n" }}\n{{- "Today Date: " + date_string + "\\n\\n" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] | trim + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n        {{- \'{"name": "\' + tool_call.name + \'", \' }}\n        {{- \'"parameters": \' }}\n        {{- tool_call.arguments | tojson }}\n        {{- "}" }}\n        {{- "<|eot_id|>" }}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n',
    'TINYLLAMA': "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}",
    'SMOLLM2': "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
}

GEMMA3_TEMPLATE = (
    "{{ bos_token }}\n{%- if messages[0]['role'] == 'system' -%}\n    {%- set first_user_prefix = messages[0]['content'] + '\\n\\n' -%}\n"
    "    {%- set loop_messages = messages[1:] -%}\n{%- else -%}\n    {%- set first_user_prefix = \"\" -%}\n"
    "    {%- set loop_messages = messages -%}\n{%- endif -%}\n{%- for message in loop_messages -%}\n"
    "    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}\n"
    "        {{ raise_exception(\"Conversation roles must alternate user/assistant/user/assistant/...\") }}\n    {%- endif -%}\n"
    "    {%- if (message['role'] == 'assistant') -%}\n        {%- set role = \"model\" -%}\n    {%- else -%}\n"
    "        {%- set role = message['role'] -%}\n    {%- endif -%}\n"
    "    {{ '<start_of_turn>' + role + '\\n' + (first_user_prefix if loop.first else \"\") }}\n"
    "    {%- if message['content'] is string -%}\n        {{ message['content'] | trim }}\n"
    "    {%- elif message['content'] is iterable -%}\n        {{ message['content'][0]['text'] | trim }}\n    {%- endif -%}\n"
    "    {{ '<end_of_turn>\\n' }}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{'<start_of_turn>model\\n'}}\n{%- endif -%}\n")
PHI3_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{{'<|system|>\\n' + message['content'] + "
    "'<|end|>\\n'}}{% elif message['role'] == 'user' %}{{'<|user|>\\n' + message['content'] + '<|end|>\\n'}}"
    "{% elif message['role'] == 'assistant' %}{{'<|assistant|>\\n' + message['content'] + '<|end|>\\n'}}{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% else %}{{ eos_token }}{% endif %}")
LLAMA2_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}"
    "{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate "
    "user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}"
    "{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}"
    "{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}"
    "{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}"
    "{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}")
MISTRAL_V02_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}"
    "{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}"
    "{% endfor %}")
MISTRAL_V03_TEMPLATE = (  # shortened: the parts that decide the layout
    "{{- bos_token }}{%- for message in loop_messages %}{%- if message[\"role\"] == \"user\" %}"
    "{%- if loop.last and system_message is defined %}{{- \"[INST] \" + system_message + \"\\n\\n\" + message[\"content\"] + \"[/INST]\" }}"
    "{%- else %}{{- \"[INST] \" + message[\"content\"] + \"[/INST]\" }}{%- endif %}{%- elif message[\"role\"] == \"assistant\" %}"
    "{{- message[\"content\"] + eos_token}}{%- endif %}{%- endfor %}")
FIXED_DATE = "05 Mar 2025"


def system(text):
    return {"role": "system", "content": text}


def user(text):
    return {"role": "user", "content": text}


def assistant(text):
    return {"role": "assistant", "content": text}


CONVERSATIONS = {
    "user only": [user("Hello")],
    "system and user": [system("Be brief."), user("Hi there")],
    "multi-turn": [user("Hi"), assistant("Hello! How can I help?"), user("What is 2+2?")],
    "system multi-turn": [system("You are a pirate."), user("Hi"), assistant("Arr!"), user("Bye")],
    "awkward whitespace": [user("  padded \n"), assistant("\n answer \n"), user("unicode: café 你好 😀\n\n")],
}
NO_SYSTEM = {k: v for k, v in CONVERSATIONS.items() if not any(m["role"] == "system" for m in v)}


def family_tokenizer(template, bos="<s>", eos="</s>"):
    """A tokenizer whose BOS/EOS texts are the given ones, carrying the given chat template."""
    meta = tiny_gguf.spm_tokenizer_meta(chat_template=template)
    meta["tokenizer.ggml.tokens"][1:3] = [bos, eos]
    return Tokenizer.from_gguf(meta)


def jinja_render(template, messages, add_generation_prompt=True, bos="", eos="", **extra):
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env.globals.update(raise_exception=raise_exception, strftime_now=lambda fmt: FIXED_DATE)
    return env.from_string(template).render(messages=messages, add_generation_prompt=add_generation_prompt, bos_token=bos,
                                            eos_token=eos, **extra)


@needs_numpy
class ChatTemplateTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(tk, "_today", return_value=FIXED_DATE)
        patcher.start()
        self.addCleanup(patcher.stop)

    def render(self, template, messages, bos="<s>", eos="</s>", **kw):
        return family_tokenizer(template, bos, eos).apply_chat_template(messages, **kw)

    def test_family_detection(self):
        expected = {"QWEN25": "chatml", "QWEN3": "chatml", "SMOLLM2": "chatml", "LLAMA32": "llama3", "TINYLLAMA": "zephyr",
                    GEMMA3_TEMPLATE: "gemma", PHI3_TEMPLATE: "phi3", LLAMA2_TEMPLATE: "llama2", MISTRAL_V02_TEMPLATE: "mistral",
                    MISTRAL_V03_TEMPLATE: "mistral", None: "chatml", "": "chatml", "{{ messages }}": "chatml"}
        for key, family in expected.items():
            template = REAL_TEMPLATES.get(key, key)
            self.assertEqual(tk.detect_chat_format(template), family, str(key)[:30])
            self.assertEqual(family_tokenizer(template).chat_format, family)

    def test_chatml_with_qwen_default_system_prompt(self):
        t = REAL_TEMPLATES["QWEN25"]
        self.assertEqual(self.render(t, [user("Hi")]), "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful "
                         "assistant.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n")
        self.assertEqual(self.render(t, [system("Be brief."), user("Hi")], add_generation_prompt=False),
                         "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n")

    def test_chatml_for_models_without_a_default_system_prompt(self):
        self.assertEqual(self.render(REAL_TEMPLATES["QWEN3"], [user("Hi")]), "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n")
        self.assertEqual(self.render(REAL_TEMPLATES["SMOLLM2"], [user("Hi")]),
                         "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n"
                         "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n")

    def test_qwen3_drops_earlier_reasoning_and_honours_thinking_prefills(self):
        t = REAL_TEMPLATES["QWEN3"]
        out = self.render(t, [user("2+2?"), assistant("<think>\nadd them\n</think>\n\n4"), user("and 3+3?")])
        self.assertEqual(out, "<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>\n<|im_start|>user\nand 3+3?"
                              "<|im_end|>\n<|im_start|>assistant\n")
        forced = t + "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n<think>\\n' }}{%- endif %}"
        self.assertTrue(self.render(forced, [user("Hi")]).endswith("<|im_start|>assistant\n<think>\n"))

    def test_qwen3_can_skip_thinking(self):
        t = REAL_TEMPLATES["QWEN3"]
        off = self.render(t, [user("Hi")], enable_thinking=False)
        self.assertEqual(off, "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        self.assertEqual(self.render(t, [user("Hi")], enable_thinking=True), self.render(t, [user("Hi")]))
        self.assertEqual(self.render(t, [user("Hi")], add_generation_prompt=False, enable_thinking=False), "<|im_start|>user\nHi<|im_end|>\n")
        self.assertNotIn("<think>", self.render(REAL_TEMPLATES["QWEN25"], [user("Hi")], enable_thinking=False))  # not a Qwen3 template
        self.assertNotIn("<think>", self.render(REAL_TEMPLATES["LLAMA32"], [user("Hi")], enable_thinking=False))
        if jinja2 is not None:
            self.assertEqual(off, jinja_render(t, [user("Hi")], True, "<s>", "</s>", enable_thinking=False))

    def test_llama3_family(self):
        plain = "{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>{% endfor %}{{ bos_token }}"
        out = self.render(plain, [system("Be brief."), user(" Hi \n")], bos="<|begin_of_text|>", eos="<|eot_id|>")
        self.assertEqual(out, "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBe brief.<|eot_id|>"
                              "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        out = self.render(REAL_TEMPLATES["LLAMA32"], [system("Be brief."), user("Hi")], bos="<|begin_of_text|>", eos="<|eot_id|>")
        self.assertEqual(out, "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n"
                              "Today Date: 05 Mar 2025\n\nBe brief.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
                              "<|start_header_id|>assistant<|end_header_id|>\n\n")
        without_date_function = REAL_TEMPLATES["LLAMA32"].replace("strftime_now", "none_such")
        self.assertIn("Today Date: 26 Jul 2024", self.render(without_date_function, [user("Hi")], bos="<|begin_of_text|>"))

    def test_the_llama3_date_is_today_by_default(self):
        self.assertRegex(TODAY(), r"^\d\d (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}$")
        self.assertTrue(TODAY().endswith(str(datetime.date.today().year)))

    def test_gemma_has_no_system_role(self):
        out = self.render(GEMMA3_TEMPLATE, [system("Be brief."), user(" Hi "), assistant("Hello"), user("Bye")], bos="<bos>", eos="<eos>")
        self.assertEqual(out, "<bos><start_of_turn>user\nBe brief.\n\nHi<end_of_turn>\n<start_of_turn>model\nHello<end_of_turn>\n"
                              "<start_of_turn>user\nBye<end_of_turn>\n<start_of_turn>model\n")

    def test_phi3(self):
        out = self.render(PHI3_TEMPLATE, [system("Be brief."), user("Hi")], bos="<s>", eos="<|endoftext|>")
        self.assertEqual(out, "<s><|system|>\nBe brief.<|end|>\n<|user|>\nHi<|end|>\n<|assistant|>\n")
        self.assertEqual(self.render(PHI3_TEMPLATE, [user("Hi")], bos="<s>", eos="<|endoftext|>", add_generation_prompt=False),
                         "<s><|user|>\nHi<|end|>\n<|endoftext|>")

    def test_zephyr_as_used_by_tinyllama(self):
        t = REAL_TEMPLATES["TINYLLAMA"]
        self.assertEqual(self.render(t, [system("Be brief."), user("Hi"), assistant("Hello"), user("Bye")]),
                         "<|system|>\nBe brief.</s>\n<|user|>\nHi</s>\n<|assistant|>\nHello</s>\n<|user|>\nBye</s>\n<|assistant|>\n")

    def test_llama2(self):
        out = self.render(LLAMA2_TEMPLATE, [system("Be brief."), user("Hi"), assistant("Hello!"), user(" Bye ")])
        self.assertEqual(out, "<s>[INST] <<SYS>>\nBe brief.\n<</SYS>>\n\nHi [/INST] Hello! </s><s>[INST] Bye [/INST]")

    def test_mistral_versions(self):
        convo = [system("Be brief."), user("Hi"), assistant("Hello!"), user("Bye")]
        self.assertEqual(self.render(MISTRAL_V02_TEMPLATE, convo),
                         "<s>[INST] Be brief.\n\nHi [/INST]Hello!</s>[INST] Bye [/INST]")
        self.assertEqual(self.render(MISTRAL_V03_TEMPLATE, convo),
                         "<s>[INST] Hi[/INST]Hello!</s>[INST] Be brief.\n\nBye[/INST]")
        self.assertEqual(self.render(MISTRAL_V02_TEMPLATE, [system("Only a system")]), "<s>[INST] Only a system [/INST]")

    @unittest.skipIf(jinja2 is None, "Jinja2 is not installed")
    def test_agrees_with_the_jinja_templates_the_models_ship(self):
        cases = [("QWEN25", CONVERSATIONS, "<s>", "</s>"), ("QWEN3", CONVERSATIONS, "<s>", "</s>"),
                 ("SMOLLM2", CONVERSATIONS, "<s>", "</s>"), ("LLAMA32", CONVERSATIONS, "<|begin_of_text|>", "<|eot_id|>"),
                 ("TINYLLAMA", CONVERSATIONS, "<s>", "</s>"), (GEMMA3_TEMPLATE, CONVERSATIONS, "<bos>", "<eos>"),
                 (PHI3_TEMPLATE, CONVERSATIONS, "<s>", "<|endoftext|>"), (LLAMA2_TEMPLATE, CONVERSATIONS, "<s>", "</s>"),
                 (MISTRAL_V02_TEMPLATE, NO_SYSTEM, "<s>", "</s>")]
        cases.append(("QWEN3", {"reasoning history": [user("2+2?"), assistant("<think>\nadd\n</think>\n\n4"), user("3+3?")]}, "<s>", "</s>"))
        for key, conversations, bos, eos in cases:
            template = REAL_TEMPLATES.get(key, key)
            for label, messages in conversations.items():
                for generation in (True, False):
                    with self.subTest(str(key)[:12], conversation=label, generation=generation):
                        want = jinja_render(template, messages, generation, bos, eos)
                        got = self.render(template, messages, bos=bos, eos=eos, add_generation_prompt=generation)
                        self.assertEqual(got, want)

    def test_message_shapes(self):
        t = REAL_TEMPLATES["QWEN3"]
        parts = {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {}}, {"type": "text", "text": "b"}]}
        self.assertEqual(self.render(t, [parts]), "<|im_start|>user\na\nb<|im_end|>\n<|im_start|>assistant\n")
        self.assertEqual(self.render(t, [{"role": "user", "content": None}]), "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n")
        self.assertEqual(self.render(t, [{"role": "developer", "content": "d"}, {"role": "tool", "content": "r"}]),
                         "<|im_start|>system\nd<|im_end|>\n<|im_start|>user\nr<|im_end|>\n<|im_start|>assistant\n")
        for bad in ([], None, ["text"], "hi"):
            with self.assertRaises(ValueError):
                self.render(t, bad)

    def test_the_bos_text_is_written_only_when_the_template_does(self):
        self.assertTrue(self.render(GEMMA3_TEMPLATE, [user("x")], bos="<bos>").startswith("<bos>"))
        self.assertFalse(self.render(REAL_TEMPLATES["TINYLLAMA"], [user("x")]).startswith("<s>"))

    def test_a_templated_prompt_encodes_with_a_single_bos(self):
        tok = family_tokenizer(GEMMA3_TEMPLATE, "<bos>", "<eos>")
        ids = tok.encode(tok.apply_chat_template([user("x")]), add_bos=True)
        self.assertEqual(ids.count(tok.bos_id), 1)
        self.assertEqual(ids[0], tok.bos_id)
        tinyllama = family_tokenizer(REAL_TEMPLATES["TINYLLAMA"])
        ids = tinyllama.encode(tinyllama.apply_chat_template([user("x")]), add_bos=tinyllama.add_bos)
        self.assertEqual(ids.count(tinyllama.bos_id), 1)  # added by the tokenizer because the template does not write it


# ---------------------------------------------------------------------------------------------------------------------
# Sampler
if np is not None:
    from duckstoken.engines.builtin.sampler import Sampler  # noqa: E402


def draw(sampler, logits, n, history=()):
    return [sampler.sample(logits, list(history)) for _ in range(n)]


@needs_numpy
class SamplerTests(unittest.TestCase):
    def test_zero_temperature_is_greedy_and_ties_go_to_the_lowest_id(self):
        s = Sampler(temperature=0)
        self.assertEqual(s.sample([0.1, 2.0, 0.3], []), 1)
        self.assertEqual(s.sample(np.array([1.0, 3.0, 3.0, 2.0], np.float32), []), 1)
        self.assertEqual(Sampler(temperature=-1).sample([0.0, 1.0], []), 1)

    def test_same_seed_same_tokens_different_seed_different_tokens(self):
        logits = np.zeros(200, np.float32)
        a = draw(Sampler(seed=7, top_k=0, top_p=1.0), logits, 60)
        b = draw(Sampler(seed=7, top_k=0, top_p=1.0), logits, 60)
        c = draw(Sampler(seed=8, top_k=0, top_p=1.0), logits, 60)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertGreater(len(set(a)), 30)
        self.assertNotEqual(draw(Sampler(seed=None, top_k=0, top_p=1.0), logits, 40), draw(Sampler(seed=None, top_k=0, top_p=1.0), logits, 40))
        self.assertNotEqual(draw(Sampler(seed=-1, top_k=0, top_p=1.0), logits, 40), draw(Sampler(seed=-1, top_k=0, top_p=1.0), logits, 40))

    def test_samples_follow_the_softmax_of_the_logits(self):
        logits = np.log(np.array([0.5, 0.3, 0.15, 0.05], np.float32))
        counts = np.bincount(draw(Sampler(temperature=1.0, top_k=0, top_p=1.0, seed=1), logits, 20000), minlength=4) / 20000
        np.testing.assert_allclose(counts, [0.5, 0.3, 0.15, 0.05], atol=0.015)
        sharper = np.bincount(draw(Sampler(temperature=0.5, top_k=0, top_p=1.0, seed=1), logits, 20000), minlength=4) / 20000
        want = np.array([0.5, 0.3, 0.15, 0.05]) ** 2
        np.testing.assert_allclose(sharper, want / want.sum(), atol=0.015)

    def test_extreme_temperatures(self):
        logits = np.array([1.0, 2.0, 3.0, 2.5], np.float32)
        self.assertEqual(set(draw(Sampler(temperature=1e-4, seed=2), logits, 100)), {2})
        self.assertEqual(len(set(draw(Sampler(temperature=1000.0, top_k=0, top_p=1.0, seed=2), logits, 200))), 4)

    def test_top_k_limits_the_candidates(self):
        logits = np.arange(50, dtype=np.float32)
        for k in (1, 3, 10):
            self.assertLessEqual(min(draw(Sampler(top_k=k, top_p=1.0, seed=3, temperature=5.0), logits, 400)), 50 - k)
            self.assertGreaterEqual(min(draw(Sampler(top_k=k, top_p=1.0, seed=3, temperature=5.0), logits, 400)), 50 - k)
        self.assertEqual(set(draw(Sampler(top_k=1, seed=3), logits, 20)), {49})
        self.assertEqual(len(set(draw(Sampler(top_k=1000, top_p=1.0, seed=3, temperature=50.0), logits, 600))), 50)

    def test_top_p_keeps_the_smallest_set_reaching_p(self):
        logits = np.log(np.array([0.1, 0.5, 0.3, 0.1], np.float32))  # sorted: 0.5, 0.3, 0.1, 0.1
        for p, allowed in ((0.4, {1}), (0.6, {1, 2}), (0.81, {1, 2, 0, 3}), (1.0, {0, 1, 2, 3})):
            seen = set(draw(Sampler(top_p=p, top_k=0, temperature=1.0, seed=4), logits, 600))
            self.assertTrue(seen <= allowed, (p, seen))
            if p >= 0.6:
                self.assertIn(2, seen)

    def test_min_p_drops_tokens_far_below_the_best(self):
        logits = np.log(np.array([0.5, 0.3, 0.15, 0.05], np.float32))
        self.assertEqual(set(draw(Sampler(min_p=0.4, top_k=0, top_p=1.0, seed=5), logits, 400)), {0, 1})
        self.assertEqual(set(draw(Sampler(min_p=0.15, top_k=0, top_p=1.0, seed=5), logits, 800)), {0, 1, 2})

    def test_repeat_penalty_divides_positive_and_multiplies_negative_logits(self):
        greedy = dict(temperature=0, repeat_penalty=1.5)
        self.assertEqual(Sampler(**greedy).sample([2.0, 1.5], [0]), 1)    # 2.0 / 1.5 = 1.33 < 1.5
        self.assertEqual(Sampler(**greedy).sample([-1.0, -1.2], [0]), 1)  # -1.0 * 1.5 = -1.5 < -1.2
        self.assertEqual(Sampler(**greedy).sample([2.0, 1.5], [1]), 0)
        self.assertEqual(Sampler(temperature=0, repeat_penalty=1.0).sample([2.0, 1.5], [0, 0]), 0)

    def test_presence_and_frequency_penalties(self):
        self.assertEqual(Sampler(temperature=0, presence_penalty=0.5).sample([1.0, 0.8], [0]), 1)
        self.assertEqual(Sampler(temperature=0, presence_penalty=0.5).sample([1.0, 0.8], [0, 0, 0]), 1)  # once, however often
        self.assertEqual(Sampler(temperature=0, presence_penalty=0.1).sample([1.0, 0.8], [0, 0, 0]), 0)
        self.assertEqual(Sampler(temperature=0, frequency_penalty=0.1).sample([1.0, 0.8], [0, 0, 0]), 1)   # 1.0 - 3 * 0.1
        self.assertEqual(Sampler(temperature=0, frequency_penalty=0.05).sample([1.0, 0.8], [0, 0, 0]), 0)  # 1.0 - 3 * 0.05
        self.assertEqual(Sampler(temperature=0, frequency_penalty=-0.5).sample([1.0, 1.2], [0]), 0)        # negative = encourage

    def test_penalties_only_look_at_the_recent_window(self):
        kw = dict(temperature=0, repeat_penalty=2.0)
        history = [1, 0]
        self.assertEqual(Sampler(repeat_last_n=2, **kw).sample([1.2, 1.0], history), 0)   # both penalized: 0.6 vs 0.5
        self.assertEqual(Sampler(repeat_last_n=1, **kw).sample([1.2, 1.0], history), 1)   # only the latest token (0) is penalized
        self.assertEqual(Sampler(repeat_last_n=-1, **kw).sample([1.2, 1.0], history), 0)  # the whole history
        self.assertEqual(Sampler(repeat_last_n=-1, **kw).sample([1.2, 1.0], [0]), 1)
        self.assertEqual(Sampler(repeat_last_n=0, **kw).sample([1.2, 1.0], [0]), 0)       # disabled
        self.assertEqual(Sampler(**kw).sample([1.2, 1.0], [99, -3]), 0)                   # ids outside the vocabulary are ignored

    def test_logits_are_not_modified(self):
        logits = np.array([1.0, 2.0, 3.0], np.float32)
        before = logits.copy()
        Sampler(repeat_penalty=2.0, presence_penalty=1.0, seed=1).sample(logits, [0, 1, 2])
        np.testing.assert_array_equal(logits, before)

    def test_masked_tokens_are_never_chosen(self):
        logits = np.array([-np.inf, 1.0, -np.inf, 0.5], np.float32)
        self.assertEqual(set(draw(Sampler(top_k=0, top_p=1.0, temperature=3.0, seed=6), logits, 200)), {1, 3})

    def test_bad_input_is_rejected(self):
        for bad in ({"temperature": float("nan")}, {"top_p": "x"}, {"repeat_penalty": 0}, {"repeat_penalty": -1.0},
                    {"min_p": float("inf")}, {"top_k": None}):
            with self.assertRaises(ValueError, msg=str(bad)):
                Sampler(**bad)
        s = Sampler()
        for logits in ([float("nan"), 1.0], [], [[1.0, 2.0]], [-np.inf, -np.inf]):
            with self.assertRaises(ValueError, msg=str(logits)):
                s.sample(logits, [])


# ---------------------------------------------------------------------------------------------------------------------
# The transformer
if np is not None:
    from duckstoken.engines.builtin import model as mdl  # noqa: E402
    from duckstoken.engines.builtin.model import ContextFull, LoadedModel  # noqa: E402


def reference_logits(path, tokens):
    """A slow, literal, double-precision forward pass (one token at a time, loops over heads and positions) of a tiny model file.
    Shares no code with the vectorised implementation."""
    with GGUFFile(path) as gf:
        meta = gf.meta
        w = {name: gf.load(name).astype(np.float64).tolist() for name in gf.tensors}
    arch = meta["general.architecture"]

    def hp(key, default=None):
        return meta.get(f"{arch}.{key}", default)

    n_layer, n_embd, n_head, n_kv = hp("block_count"), hp("embedding_length"), hp("attention.head_count"), hp("attention.head_count_kv")
    hd = hp("attention.key_length", n_embd // n_head)
    n_rot, base, eps = hp("rope.dimension_count", hd), hp("rope.freq_base"), hp("attention.layer_norm_rms_epsilon")
    scale = hp("rope.scaling.factor", 1.0) if hp("rope.scaling.type") == "linear" else 1.0
    factors = w.get("rope_freqs.weight")
    pairs_adjacent = arch == "llama"

    def matvec(matrix, vec):
        return [sum(a * b for a, b in zip(row, vec)) for row in matrix]

    def rms(vec, weight):
        r = 1.0 / math.sqrt(sum(x * x for x in vec) / len(vec) + eps)
        return [x * r * g for x, g in zip(vec, weight)]

    def rotate(vec, pos):
        out = list(vec)
        for i in range(n_rot // 2):
            theta = pos / scale * base ** (-2.0 * i / n_rot) / (factors[i] if factors else 1.0)
            c, s = math.cos(theta), math.sin(theta)
            a_at, b_at = (2 * i, 2 * i + 1) if pairs_adjacent else (i, i + n_rot // 2)
            out[a_at], out[b_at] = vec[a_at] * c - vec[b_at] * s, vec[b_at] * c + vec[a_at] * s
        return out

    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    x = None
    for pos, token in enumerate(tokens):
        x = list(w["token_embd.weight"][token])
        for layer in range(n_layer):
            b = f"blk.{layer}."
            h = rms(x, w[b + "attn_norm.weight"])
            q, k, v = (matvec(w[b + f"attn_{n}.weight"], h) for n in "qkv")
            if b + "attn_q.bias" in w:
                q, k, v = ([a + c for a, c in zip(vec, w[b + f"attn_{n}.bias"])] for vec, n in ((q, "q"), (k, "k"), (v, "v")))
            q_heads = [q[i * hd:(i + 1) * hd] for i in range(n_head)]
            k_heads = [k[i * hd:(i + 1) * hd] for i in range(n_kv)]
            if b + "attn_q_norm.weight" in w:
                q_heads = [rms(head, w[b + "attn_q_norm.weight"]) for head in q_heads]
                k_heads = [rms(head, w[b + "attn_k_norm.weight"]) for head in k_heads]
            q_heads = [rotate(head, pos) for head in q_heads]
            keys[layer].append([rotate(head, pos) for head in k_heads])
            values[layer].append([v[i * hd:(i + 1) * hd] for i in range(n_kv)])
            attended = []
            for head in range(n_head):
                kv = head // (n_head // n_kv)
                scores = [sum(a * c for a, c in zip(q_heads[head], keys[layer][t][kv])) / math.sqrt(hd) for t in range(pos + 1)]
                top = max(scores)
                weights = [math.exp(sc - top) for sc in scores]
                total = sum(weights)
                attended += [sum(weights[t] * values[layer][t][kv][d] for t in range(pos + 1)) / total for d in range(hd)]
            x = [a + c for a, c in zip(x, matvec(w[b + "attn_output.weight"], attended))]
            h = rms(x, w[b + "ffn_norm.weight"])
            gate, up = matvec(w[b + "ffn_gate.weight"], h), matvec(w[b + "ffn_up.weight"], h)
            act = [g / (1.0 + math.exp(-g)) * u for g, u in zip(gate, up)]
            x = [a + c for a, c in zip(x, matvec(w[b + "ffn_down.weight"], act))]
    return matvec(w.get("output.weight", w["token_embd.weight"]), rms(x, w["output_norm.weight"]))


_MODELS: dict = {}


def tiny_model(name: str, **kw):
    """Write (once) and return a tiny model file: `name` identifies the variant."""
    if name not in _MODELS:
        _MODELS[name] = tiny_gguf.write_tiny_llama_gguf(tmp_path(f"{name}.gguf"), **kw)
    return _MODELS[name]


VARIANTS = {
    "llama": dict(arch="llama"),
    "llama-quantized": dict(arch="llama", quant="q4_0", seed=2),
    "qwen2": dict(arch="qwen2", seed=3),
    "qwen2-q8": dict(arch="qwen2", quant="q8_0", seed=4),
    "qwen3": dict(arch="qwen3", head_dim=16, seed=5),
    "mha": dict(arch="llama", n_head=4, n_kv_head=4, seed=6),
    "mqa": dict(arch="qwen2", n_head=4, n_kv_head=1, seed=7),
    "tied": dict(arch="llama", tie_embeddings=True, seed=8),
    "partial-rope": dict(arch="llama", extra_meta={"llama.rope.dimension_count": 4}, seed=9),
    "partial-rope-neox": dict(arch="qwen2", extra_meta={"qwen2.rope.dimension_count": 4}, seed=10),
    "rope-factors": dict(arch="llama", rope_freqs=np.array([1.0, 1.0, 2.5, 8.0]) if np is not None else None, rope_base=500000.0, seed=11),
    "linear-scaling": dict(arch="llama", extra_meta={"llama.rope.scaling.type": "linear", "llama.rope.scaling.factor": 4.0}, seed=12),
}
PROMPT = [3, 77, 150, 12, 200, 45, 9, 301, 66, 14]


@needs_numpy
class TransformerTests(unittest.TestCase):
    def test_matches_a_slow_reference_implementation(self):
        for name, kw in VARIANTS.items():
            with self.subTest(name):
                tm = tiny_model(name, **kw)
                model = load_model(tm.path, max_ctx=64)
                try:
                    got = model.new_session().feed(PROMPT)
                finally:
                    model.close()
                want = np.array(reference_logits(tm.path, PROMPT))
                self.assertEqual(got.dtype, np.float32)
                self.assertEqual(got.shape, (len(tm.tokens),))
                np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)

    def test_the_comparison_notices_a_wrong_rope_pairing(self):
        tm = tiny_model("llama", **VARIANTS["llama"])
        model = load_model(tm.path, max_ctx=64)
        model.cfg = dataclasses.replace(model.cfg, neox=True)  # rotate the wrong pairs on purpose
        got = model.new_session().feed(PROMPT)
        self.assertGreater(np.abs(got - np.array(reference_logits(tm.path, PROMPT))).max(), 1e-3)

    def test_one_token_at_a_time_equals_all_at_once(self):
        for name in ("llama", "qwen2", "qwen3", "mqa"):
            with self.subTest(name):
                model = load_model(tiny_model(name, **VARIANTS[name]).path, max_ctx=64)
                batched = model.new_session().feed(PROMPT)
                s = model.new_session()
                for t in PROMPT:
                    stepped = s.feed([t])
                self.assertEqual(s.n_past, len(PROMPT))
                np.testing.assert_allclose(stepped, batched, rtol=1e-4, atol=1e-5)

    def test_prompt_chunking_does_not_change_the_result(self):
        model = load_model(tiny_model("qwen2", **VARIANTS["qwen2"]).path, max_ctx=64)
        whole = model.new_session().feed(PROMPT * 3)
        with mock.patch.object(mdl, "BATCH", 4):
            s = model.new_session()
            chunked = s.feed(PROMPT * 3)
        self.assertEqual(s.n_past, 30)
        np.testing.assert_allclose(chunked, whole, rtol=1e-4, atol=1e-5)

    def test_truncate_reuses_the_cached_prefix(self):
        model = load_model(tiny_model("llama", **VARIANTS["llama"]).path, max_ctx=64)
        s = model.new_session()
        s.feed(PROMPT)
        s.truncate(6)
        self.assertEqual(s.n_past, 6)
        after = s.feed([5, 6, 7])
        fresh = model.new_session().feed(PROMPT[:6] + [5, 6, 7])
        np.testing.assert_allclose(after, fresh, rtol=1e-4, atol=1e-5)
        s.truncate(0)
        np.testing.assert_allclose(s.feed(PROMPT), model.new_session().feed(PROMPT), rtol=1e-6)
        for bad in (-1, s.n_past + 1):
            with self.assertRaises(ValueError):
                s.truncate(bad)

    def test_reset_and_independent_sessions(self):
        model = load_model(tiny_model("llama", **VARIANTS["llama"]).path, max_ctx=64)
        a, b = model.new_session(), model.new_session()
        first = a.feed(PROMPT)
        b.feed([1, 2, 3])  # another conversation does not disturb a
        np.testing.assert_allclose(a.feed([4]), model.new_session().feed(PROMPT + [4]), rtol=1e-4, atol=1e-5)
        a.reset()
        self.assertEqual(a.n_past, 0)
        np.testing.assert_allclose(a.feed(PROMPT), first, rtol=1e-6)

    def test_the_context_window_is_enforced(self):
        model = load_model(tiny_model("llama", **VARIANTS["llama"]).path, max_ctx=16)
        s = model.new_session()
        s.feed(list(range(16)))
        with self.assertRaises(ContextFull):
            s.feed([1])
        s.truncate(15)
        s.feed([1])
        s.reset()
        with self.assertRaises(ContextFull):
            s.feed(list(range(17)))
        self.assertEqual(s.n_past, 0)

    def test_invalid_token_ids_are_rejected(self):
        model = load_model(tiny_model("llama", **VARIANTS["llama"]).path, max_ctx=16)
        s = model.new_session()
        for bad in ([], [-1], [model.vocab_size], [[1, 2]], 5, ["a"], [2 ** 70], None):
            with self.assertRaises(ValueError, msg=str(bad)):
                s.feed(bad)
        self.assertEqual(s.n_past, 0)

    def test_greedy_generation_is_deterministic(self):
        model = load_model(tiny_model("qwen3", **VARIANTS["qwen3"]).path, max_ctx=64)

        def run():
            s, sampler, history = model.new_session(), Sampler(temperature=0), list(PROMPT)
            logits = s.feed(history)
            for _ in range(12):
                history.append(sampler.sample(logits, history))
                logits = s.feed([history[-1]])
            return history

        self.assertEqual(run(), run())

    def test_context_is_clamped_to_what_the_model_supports(self):
        tm = tiny_gguf.write_tiny_llama_gguf(tmp_path("short-ctx.gguf"), context_length=48)
        model = load_model(tm.path, max_ctx=4096)
        self.assertEqual((model.max_ctx, model.context_length), (48, 48))
        self.assertEqual(load_model(tm.path, max_ctx=32).max_ctx, 32)
        with self.assertRaises(ValueError):
            load_model(tm.path, max_ctx=2)

    def test_model_facts(self):
        tm = tiny_model("qwen2", **VARIANTS["qwen2"])
        model = load_model(tm.path, max_ctx=64)
        self.assertEqual((model.name, model.arch, model.vocab_size), ("tiny", "qwen2", len(tm.tokens)))
        self.assertEqual(model.n_params, sum(a.size for a in tm.tensors.values()))
        self.assertEqual((model.weight_dtype, model.kv_dtype), ("float32", "float32"))
        self.assertEqual(model.tokenizer.vocab_size, len(tm.tokens))
        self.assertIsNotNone(model.tokenizer.eos_ids)
        self.assertTrue(model.new_session().feed([1]).shape == (model.vocab_size,))

    def test_progress_is_reported_from_start_to_finish(self):
        calls = []
        load_model(tiny_model("llama", **VARIANTS["llama"]).path, max_ctx=16, progress=lambda f, m: calls.append((f, m)))
        fractions = [f for f, _ in calls]
        self.assertGreater(len(calls), 5)
        self.assertEqual(fractions, sorted(fractions))
        self.assertEqual(fractions[-1], 1.0)
        self.assertTrue(all(0 <= f <= 1 and isinstance(m, str) and m for f, m in calls))

    def test_every_supported_format_loads_and_runs(self):
        for quant in tiny_gguf.BLOCK:
            with self.subTest(quant):
                tm = tiny_gguf.write_tiny_llama_gguf(tmp_path(f"fmt-{quant}.gguf"), quant=quant, n_embd=256, n_head=4, n_kv_head=2,
                                                     n_ff=256, n_layer=1, seed=21)
                model = load_model(tm.path, max_ctx=16)
                logits = model.new_session().feed([1, 2, 3])
                self.assertTrue(np.isfinite(logits).all())
                want = np.array(reference_logits(tm.path, [1, 2, 3]))
                np.testing.assert_allclose(logits, want, rtol=1e-3, atol=1e-3)

    def test_quantized_models_track_the_float_model(self):
        base = tiny_gguf.write_tiny_llama_gguf(tmp_path("q-base.gguf"), n_embd=256, n_head=8, n_kv_head=4, n_ff=256, n_layer=2, seed=22)
        want = load_model(base.path, max_ctx=16).new_session().feed([5, 6, 7, 8])
        for quant, floor in (("q8_0", 0.999), ("q6_k", 0.99), ("q4_k", 0.9)):
            tm = tiny_gguf.write_tiny_llama_gguf(tmp_path(f"q-{quant}.gguf"), quant=quant, n_embd=256, n_head=8, n_kv_head=4, n_ff=256,
                                                 n_layer=2, seed=22)
            got = load_model(tm.path, max_ctx=16).new_session().feed([5, 6, 7, 8])
            cosine = float(got @ want / np.linalg.norm(got) / np.linalg.norm(want))
            self.assertGreater(cosine, floor, quant)


@needs_numpy
class InspectTests(unittest.TestCase):
    def test_reports_what_a_file_needs_without_loading_it(self):
        tm = tiny_model("qwen2-q8", **VARIANTS["qwen2-q8"])
        info = mdl.inspect_model(tm.path)
        resident = sum(a.size for n, a in tm.tensors.items() if n != "token_embd.weight")  # untied: the embedding table stays on disk
        self.assertEqual(info, {"name": "tiny", "arch": "qwen2", "n_params": sum(a.size for a in tm.tensors.values()),
                                "context_length": 256, "vocab_size": len(tm.tokens), "formats": ["F32", "Q8_0"], "chat_format": "chatml",
                                "ram_gb": {"float32": round(resident * 4 / 2**30, 2), "bfloat16": round(resident * 2 / 2**30, 2)}})
        self.assertEqual(mdl.SUPPORTED_ARCHITECTURES, ("llama", "qwen2", "qwen3"))

    def test_refuses_what_loading_would_refuse(self):
        mamba = tiny_gguf.write_tiny_llama_gguf(tmp_path("inspect-mamba.gguf"), arch="mamba")
        with self.assertRaisesRegex(UnsupportedModel, "llama.cpp"):
            mdl.inspect_model(mamba.path)
        with self.assertRaises(UnsupportedModel):
            mdl.inspect_model(tmp_path("inspect-missing.gguf"))
        with mock.patch.object(mdl, "MAX_PARAMS", 10):
            with self.assertRaisesRegex(UnsupportedModel, "billion"):
                mdl.inspect_model(tiny_model("llama", **VARIANTS["llama"]).path)

    def test_the_file_is_released_afterwards(self):
        tm = tiny_gguf.write_tiny_llama_gguf(tmp_path("inspect-release.gguf"))
        mdl.inspect_model(tm.path)
        os.remove(tm.path)


@needs_numpy
class WeightStorageTests(unittest.TestCase):
    def test_bfloat16_weights_track_float32(self):
        for name in ("llama", "qwen3", "tied"):
            with self.subTest(name):
                path = tiny_model(name, **VARIANTS[name]).path
                full = load_model(path, max_ctx=32, weight_dtype="float32")
                half = load_model(path, max_ctx=32, weight_dtype="bfloat16")
                self.assertEqual(half.weight_dtype, "bfloat16")
                self.assertEqual(half.head.w.dtype, np.uint16)
                want, got = full.new_session().feed(PROMPT), half.new_session().feed(PROMPT)
                self.assertLess(np.abs(got - want).max(), 0.03 * np.abs(want).max())  # bfloat16 keeps about 3 significant digits
                self.assertGreater(float(got @ want / np.linalg.norm(got) / np.linalg.norm(want)), 0.999)

    def test_bfloat16_embeddings_are_widened_exactly(self):
        model = load_model(tiny_model("tied", **VARIANTS["tied"]).path, max_ctx=32, weight_dtype="bfloat16")
        rows = model.head.rows(np.array([0, 5, 5]))
        self.assertEqual(rows.dtype, np.float32)
        np.testing.assert_array_equal(rows[1], rows[2])
        np.testing.assert_allclose(rows[0], tiny_model("tied", **VARIANTS["tied"]).tensors["token_embd.weight"][0], rtol=2 ** -8)

    def test_float16_cache_tracks_the_float32_cache(self):
        path = tiny_model("qwen2", **VARIANTS["qwen2"]).path
        full = load_model(path, max_ctx=32, kv_dtype="float32")
        half = load_model(path, max_ctx=32, kv_dtype="float16")
        self.assertEqual(half.new_session().k.dtype, np.float16)
        np.testing.assert_allclose(half.new_session().feed(PROMPT), full.new_session().feed(PROMPT), rtol=5e-3, atol=5e-3)

    def test_auto_choice_follows_the_available_memory(self):
        tm = tiny_model("llama", **VARIANTS["llama"])
        n_params = sum(a.size for a in tm.tensors.values())
        for free, expected in ((0, "float32"), (int(n_params * 4 / 0.55), "float32"), (int(n_params * 2.5 / 0.6), "bfloat16")):
            with mock.patch.object(mdl, "_free_ram_bytes", return_value=free):
                self.assertEqual(load_model(tm.path, max_ctx=8).weight_dtype, expected, free)
        with mock.patch.object(mdl, "_free_ram_bytes", return_value=n_params // 2):
            with self.assertRaisesRegex(UnsupportedModel, "RAM"):
                load_model(tm.path, max_ctx=8)
            self.assertEqual(load_model(tm.path, max_ctx=8, weight_dtype="float32").weight_dtype, "float32")  # forced

    def test_large_caches_default_to_float16(self):
        tm = tiny_model("llama", **VARIANTS["llama"])
        with mock.patch.object(mdl, "KV_FLOAT32_LIMIT", 1024):
            self.assertEqual(load_model(tm.path, max_ctx=64).kv_dtype, "float16")
        self.assertEqual(load_model(tm.path, max_ctx=64).kv_dtype, "float32")

    def test_bad_options(self):
        path = tiny_model("llama", **VARIANTS["llama"]).path
        for kw in ({"weight_dtype": "float16"}, {"kv_dtype": "bfloat16"}, {"weight_dtype": "int8"}):
            with self.assertRaises(ValueError, msg=str(kw)):
                load_model(path, **kw)

    def test_thread_limits_are_applied_and_restored(self):
        events = []

        class Limits:
            def __init__(self, limits=None, user_api=None):
                events.append(("limit", limits, user_api))

            def restore_original_limits(self):
                events.append(("restore",))

        fake = types.ModuleType("threadpoolctl")
        fake.threadpool_limits = Limits
        path = tiny_model("llama", **VARIANTS["llama"]).path
        with mock.patch.dict(sys.modules, {"threadpoolctl": fake}):
            load_model(path, max_ctx=16, n_threads=0).close()
            self.assertEqual(events, [])
            model = load_model(path, max_ctx=16, n_threads=3)
            self.assertEqual(events, [("limit", 3, "blas")])
            model.close()
            model.close()
            self.assertEqual(events[1:], [("restore",)])
        with mock.patch.dict(sys.modules, {"threadpoolctl": None}):
            load_model(path, max_ctx=16, n_threads=3).close()  # not installed: NumPy keeps its own default


@needs_numpy
class EmbeddingStorageTests(unittest.TestCase):
    def test_tied_embeddings_share_one_matrix(self):
        model = load_model(tiny_model("tied", **VARIANTS["tied"]).path, max_ctx=16)
        self.assertTrue(model.tied)
        self.assertIsNone(model._gf)

    def test_an_identical_output_matrix_counts_as_tied(self):
        tm = tiny_gguf.write_tiny_llama_gguf(tmp_path("dup-out.gguf"), seed=31, quant="q8_0",
                                             mutate=lambda w: w.update({"output.weight": w["token_embd.weight"].copy()}))
        model = load_model(tm.path, max_ctx=16)
        self.assertTrue(model.tied)
        self.assertEqual(model.n_params, sum(a.size for n, a in tm.tensors.items() if n != "output.weight"))
        np.testing.assert_allclose(model.new_session().feed([1, 2]), np.array(reference_logits(tm.path, [1, 2])), rtol=2e-4, atol=2e-4)

    def test_a_separate_output_matrix_keeps_the_embedding_table_on_disk(self):
        tm = tiny_gguf.write_tiny_llama_gguf(tmp_path("sep-out.gguf"), seed=32)
        model = load_model(tm.path, max_ctx=16)
        self.assertFalse(model.tied)
        self.assertIsNotNone(model._gf)
        np.testing.assert_allclose(model.new_session().feed([1, 2]), np.array(reference_logits(tm.path, [1, 2])), rtol=2e-4, atol=2e-4)
        model.close()
        model.close()
        os.remove(tm.path)  # the model file is released (this fails on Windows if it were still mapped)

    def test_padding_rows_of_the_vocabulary_cannot_be_sampled(self):
        def pad(w):
            for name in ("token_embd.weight", "output.weight"):
                w[name] = np.concatenate([w[name], np.ones((5, w[name].shape[1]), np.float32)])

        tm = tiny_gguf.write_tiny_llama_gguf(tmp_path("padded.gguf"), seed=33, mutate=pad)
        model = load_model(tm.path, max_ctx=16)
        n = len(tm.tokens)
        logits = model.new_session().feed([1, 2])
        self.assertEqual(model.vocab_size, n + 5)
        self.assertTrue(np.isfinite(logits[:n]).all())
        self.assertTrue(np.all(logits[n:] == -np.inf))
        self.assertLess(Sampler(temperature=2.0, top_k=0, top_p=1.0, seed=1).sample(logits, []), n)


@needs_numpy
class UnsupportedModelTests(unittest.TestCase):
    def load_error(self, name, **kw):
        with self.assertRaises(UnsupportedModel) as cm:
            load_model(tiny_gguf.write_tiny_llama_gguf(tmp_path(f"{name}.gguf"), **kw).path, max_ctx=16)
        return str(cm.exception)

    def test_other_architectures_point_at_llama_cpp(self):
        message = self.load_error("mamba", arch="mamba")
        self.assertIn("mamba", message)
        self.assertIn("llama.cpp", message)

    def test_mixture_of_experts_models_are_refused_honestly(self):
        for name, kw in (("moe-llama", dict(arch="llama", extra_meta={"llama.expert_count": 8})),
                         ("qwen3moe", dict(arch="qwen3moe")), ("gpt-oss", dict(arch="gpt-oss"))):
            message = self.load_error(name, **kw)
            self.assertIn("mixture-of-experts", message, name)
            self.assertIn("llama.cpp", message)

    def test_unsupported_rope_scaling(self):
        for kind in ("yarn", "longrope"):
            message = self.load_error(f"rope-{kind}", extra_meta={"llama.rope.scaling.type": kind})
            self.assertIn(kind, message)

    def test_missing_wrong_shaped_and_unreadable_tensors(self):
        self.assertIn("blk.1.ffn_up.weight", self.load_error("no-tensor", mutate=lambda w: w.pop("blk.1.ffn_up.weight")))
        self.assertIn("shape", self.load_error("bad-shape", mutate=lambda w: w.update({"blk.0.attn_q.weight": w["blk.0.attn_q.weight"][:, :16]})))
        self.assertIn("embedding", self.load_error("no-embd", mutate=lambda w: w.pop("token_embd.weight")))
        tm = tiny_gguf.write_tiny_llama_gguf(tmp_path("iq2.gguf"))
        tensors = [(n, a.shape, 0, a.tobytes()) for n, a in tm.tensors.items()]
        tensors[5] = (tensors[5][0], tensors[5][1], 16, b"\0" * 8)  # IQ2_XXS
        tiny_gguf.write_gguf(tmp_path("iq2.gguf"), tm.meta, tensors)
        with self.assertRaisesRegex(UnsupportedModel, "IQ2_XXS"):
            load_model(tmp_path("iq2.gguf"))

    def test_absurd_metadata_is_rejected_quickly(self):
        for key, value in (("block_count", 10 ** 9), ("embedding_length", 2 ** 40), ("attention.head_count", 0), ("context_length", -5),
                           ("feed_forward_length", True)):
            tm = tiny_gguf.write_tiny_llama_gguf(tmp_path("absurd.gguf"), extra_meta={f"llama.{key}": value})
            start = time.perf_counter()
            with self.assertRaisesRegex(UnsupportedModel, key.split(".")[-1], msg=key):
                LoadedModel.load(tm.path)
            self.assertLess(time.perf_counter() - start, 2.0)

    def test_variants_with_extra_tensors_are_refused_instead_of_answering_wrongly(self):
        message = self.load_error("extra-tensor", mutate=lambda w: w.update({"blk.0.attn_q.bias": np.zeros(32, np.float32)}))
        self.assertIn("blk.0.attn_q.bias", message)
        self.assertIn("llama.cpp", message)
        quiet = tiny_gguf.write_tiny_llama_gguf(tmp_path("old-rot.gguf"), mutate=lambda w: w.update({"blk.0.attn_rot_embd.weight": np.zeros(4, np.float32)}))
        load_model(quiet.path, max_ctx=16)  # llama.cpp's early converters stored a rotary table nobody needs

    def test_missing_metadata(self):
        tm = tiny_gguf.write_tiny_llama_gguf(tmp_path("no-meta.gguf"))
        meta = {k: v for k, v in tm.meta.items() if k != "llama.block_count"}
        tiny_gguf.write_gguf(tmp_path("no-meta.gguf"), meta, [(n, a.shape, 0, a.tobytes()) for n, a in tm.tensors.items()])
        with self.assertRaisesRegex(UnsupportedModel, "block_count"):
            load_model(tmp_path("no-meta.gguf"))

    def test_models_too_big_for_the_engine(self):
        path = tiny_model("llama", **VARIANTS["llama"]).path
        with mock.patch.object(mdl, "MAX_PARAMS", 1000):
            with self.assertRaisesRegex(UnsupportedModel, "parameters"):
                load_model(path)

    def test_the_tokenizer_cannot_exceed_the_embedding_table(self):
        message = self.load_error("small-embd", mutate=lambda w: w.update({"token_embd.weight": w["token_embd.weight"][:100],
                                                                            "output.weight": w["output.weight"][:100]}))
        self.assertIn("tokens", message)

    def test_garbage_and_missing_files(self):
        bad = tmp_path("garbage.gguf")
        bad.write_bytes(b"not a model at all" * 10)
        for path in (bad, tmp_path("does-not-exist.gguf")):
            with self.assertRaises(UnsupportedModel):
                load_model(path)

    def test_a_failed_load_leaves_no_file_open(self):
        tm = tiny_gguf.write_tiny_llama_gguf(tmp_path("fail-late.gguf"), extra_meta={"llama.rope.scaling.type": "yarn"})
        with self.assertRaises(UnsupportedModel):
            load_model(tm.path)
        os.remove(tm.path)


@unittest.skipUnless(os.environ.get("DUCKSTOKEN_TEST_GGUF"), "set DUCKSTOKEN_TEST_GGUF to a small instruct GGUF to run this")
class RealModelTests(unittest.TestCase):
    def test_greedy_answer_is_sensible(self):
        model = load_model(os.environ["DUCKSTOKEN_TEST_GGUF"], max_ctx=256)
        self.addCleanup(model.close)
        tok, s, sampler = model.tokenizer, model.new_session(), Sampler(temperature=0)
        history = tok.encode("The capital of France is", add_bos=tok.add_bos)
        logits, out = s.feed(history), []
        for _ in range(8):
            out.append(sampler.sample(logits, history))
            history.append(out[-1])
            logits = s.feed([out[-1]])
        self.assertIn("Paris", tok.decode(out))


if __name__ == "__main__":
    unittest.main()
