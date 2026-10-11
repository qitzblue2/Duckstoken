"""Built-in engine core: GGUF reading and dequantization, tokenizers, chat templates, the transformer and the sampler."""
import math
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("DUCKSTOKEN_HOME", tempfile.mkdtemp(prefix="duck-builtin-"))

try:
    import numpy as np
except ImportError:  # pragma: no cover - the whole module is skipped below
    np = None

import tiny_gguf  # noqa: E402

if np is not None:
    from duckstoken.engines.builtin import gguf_weights as gw  # noqa: E402
    from duckstoken.engines.builtin.gguf_weights import GGUFFile, UnsupportedModel  # noqa: E402

needs_numpy = unittest.skipIf(np is None, "NumPy is not installed")
TMP = tempfile.TemporaryDirectory(prefix="duck-core-")


def tmp_path(name: str) -> Path:
    return Path(TMP.name) / name


def tearDownModule():
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
                got = gw._TYPES[tiny_gguf.TYPE_ID[qtype]].dequantize(blocks)
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

    def test_load_to_float16_and_into_a_buffer(self):
        x = np.random.default_rng(5).standard_normal((3, 64)).astype(np.float32)
        path = tmp_path("f16.gguf")
        write_one_tensor(path, x, "q8_0")
        with GGUFFile(path) as f:
            full = f.load("w")
            half = f.load("w", "float16")
            buf = np.zeros((3, 64), np.float32)
            self.assertIs(f.load("w", out=buf), buf)
            self.assertEqual(half.dtype, np.float16)
            np.testing.assert_array_equal(half, full.astype(np.float16))
            np.testing.assert_array_equal(buf, full)
            with self.assertRaises(ValueError):
                f.load("w", out=np.zeros((3, 63), np.float32))
            with self.assertRaises(ValueError):
                f.load("w", "int8")

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
        evil_array = head + tiny_gguf._s("k") + struct.pack("<II", 9, 8) + struct.pack("<Q", 1 << 60)
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


if __name__ == "__main__":
    unittest.main()
