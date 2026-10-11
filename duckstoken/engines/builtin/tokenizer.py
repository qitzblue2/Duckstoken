"""Tokenizers (byte-level BPE and SentencePiece) and chat templates for the built-in engine, built from GGUF metadata.

Pure standard library. The BPE pre-tokenizer patterns are the ones llama.cpp uses; Python's `re` has no `\\p{L}`, so the Unicode
classes are generated from `unicodedata` once and substituted into the patterns.
"""

from __future__ import annotations

import codecs
import datetime
import functools
import heapq
import re
import unicodedata

from .gguf_weights import UnsupportedModel

NORMAL, UNKNOWN, CONTROL, USER_DEFINED, UNUSED, BYTE = 1, 2, 3, 4, 5, 6  # GGUF token types
SEPARATOR = "▁"  # SentencePiece's visible space
_MAX_CACHE = 50_000
_END_OF_TURN = {"<|im_end|>", "<|eot_id|>", "<|eom_id|>", "<|end|>", "<end_of_turn>", "<|endoftext|>", "<|end_of_text|>", "<eos>"}


# ---------------------------------------------------------------------------------------------------- Unicode classes
_WHITESPACE = "\\t\\n\\x0b\\x0c\\r \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000"  # Unicode White_Space


@functools.lru_cache(maxsize=1)
def _unicode_classes() -> tuple[str, str]:
    """Bodies of character classes for the Unicode categories L* and N*, as ranges of \\U escapes."""
    runs: dict[str, list[list[int]]] = {"L": [], "N": []}
    for cp in range(0x40000):  # planes 0-3 hold every assigned letter and number
        major = unicodedata.category(chr(cp))[0]
        if major in runs:
            r = runs[major]
            if r and r[-1][1] == cp - 1:
                r[-1][1] = cp
            else:
                r.append([cp, cp])
    return tuple("".join(f"\\U{a:08x}" if a == b else f"\\U{a:08x}-\\U{b:08x}" for a, b in runs[k]) for k in "LN")  # type: ignore[return-value]


def _translate(pattern: str) -> str:
    """Rewrite a PCRE-style pattern using \\p{L}, \\p{N}, \\s and \\S into plain `re` syntax with explicit classes."""
    letters, numbers = _unicode_classes()
    out, i, in_class = [], 0, False
    while i < len(pattern):
        c = pattern[i]
        if c == "\\":
            kind = pattern[i + 1]
            if kind == "p":
                end = pattern.index("}", i)
                body = {"L": letters, "N": numbers}[pattern[i + 3:end]]
                out.append(body if in_class else f"[{body}]")
                i = end + 1
                continue
            if kind in "sS":
                out.append(_WHITESPACE if in_class else ("[" if kind == "s" else "[^") + _WHITESPACE + "]")
                i += 2
                continue
            out.append(pattern[i:i + 2])
            i += 2
            continue
        in_class = (c == "[") or (in_class and c != "]")
        out.append(c)
        i += 1
    return "".join(out)


_CONTRACTIONS = "'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]"
_GPT2 = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
_QWEN2 = rf"(?:{_CONTRACTIONS})|[^\r\n\p{{L}}\p{{N}}]?\p{{L}}+|\p{{N}}| ?[^\s\p{{L}}\p{{N}}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
_LLAMA3 = _QWEN2.replace(r"|\p{N}|", r"|\p{N}{1,3}|")
# tokenizer.ggml.pre -> patterns applied one after the other, each splitting the pieces left by the previous one
PRE_TOKENIZERS = {"default": [_GPT2], "gpt-2": [_GPT2], "qwen2": [_QWEN2], "llama-bpe": [_LLAMA3], "llama3": [_LLAMA3],
                  "llama-v3": [_LLAMA3], "smollm": [r"\p{N}", _GPT2]}


@functools.lru_cache(maxsize=None)
def _compiled(patterns: tuple[str, ...]) -> tuple:
    return tuple(re.compile(_translate(p)) for p in patterns)


def _split(text: str, regexes) -> list[str]:
    pieces = [text]
    for rx in regexes:
        parts = []
        for piece in pieces:
            pos = 0
            for m in rx.finditer(piece):
                if m.start() > pos:
                    parts.append(piece[pos:m.start()])
                parts.append(m.group())
                pos = m.end()
            if pos < len(piece):
                parts.append(piece[pos:])
        pieces = parts
    return pieces


@functools.lru_cache(maxsize=1)
def _byte_unicode() -> tuple[dict[int, str], dict[str, int]]:
    """GPT-2's reversible byte <-> printable-character table."""
    keep = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    table, extra = {b: chr(b) for b in keep}, 0
    for b in range(256):
        if b not in table:
            table[b] = chr(256 + extra)
            extra += 1
    return table, {c: b for b, c in table.items()}


# -------------------------------------------------------------------------------------------------------- chat templates
def _text(content) -> str:
    """OpenAI message content (string, list of parts, or null) -> plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p["text"] for p in content if isinstance(p, dict) and isinstance(p.get("text"), str))
    return str(content)


def _messages(messages: list[dict]) -> list[tuple[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    roles = {"system": "system", "developer": "system", "user": "user", "assistant": "assistant"}
    out = []
    for m in messages:
        if not isinstance(m, dict):
            raise ValueError("each message must be an object with a role and content")
        out.append((roles.get(m.get("role"), "user"), _text(m.get("content"))))  # tool results are shown to the model as user turns
    return out


def detect_chat_format(template: str | None) -> str:
    """The prompt family of a GGUF `tokenizer.chat_template` (a Jinja string we never execute), recognised by its signature."""
    t = template or ""
    if "<|im_start|>" in t:
        return "chatml"
    if "<|start_header_id|>" in t:
        return "llama3"
    if "<start_of_turn>" in t:
        return "gemma"
    if "[INST]" in t:
        return "llama2" if "<<SYS>>" in t else "mistral"
    if "<|assistant|>" in t:
        return "phi3" if "<|end|>" in t else "zephyr"
    return "chatml"


def _today() -> str:
    d = datetime.date.today()
    return f"{d.day:02d} {('Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec').split()[d.month - 1]} {d.year}"



class Tokenizer:
    """Common behaviour: special-token handling, decoding, streaming and chat templates. Build one with `from_gguf`."""

    bos_id: int | None
    eos_id: int | None       # the end-of-sequence token (the one chat templates write); eos_ids holds every id that stops generation
    eos_ids: set[int]
    vocab_size: int
    chat_format: str
    add_bos: bool  # whether plain text prompts should start with the BOS token

    def __init__(self, tokens: list[str], types: list[int], bos_id: int | None, eos_id: int | None, eos_ids: set[int],
                 add_bos: bool, chat_template: str | None, unk_id: int | None):
        self.tokens, self.types = tokens, types
        self.vocab_size = len(tokens)
        self.bos_id, self.eos_id, self.eos_ids, self.add_bos, self.unk_id = bos_id, eos_id, eos_ids, add_bos, unk_id
        self.chat_template = chat_template or ""
        self.chat_format = detect_chat_format(chat_template)
        self._pieces: list[bytes | None] = [None] * len(tokens)
        self._special_pieces: list[bytes | None] = [None] * len(tokens)
        self._partitions: dict[bool, re.Pattern | None] = {}
        self._token_ids = {t: i for i, t in reversed(list(enumerate(tokens)))}  # the first of any duplicates wins
        self._strip_first_space = False

    # ------------------------------------------------------------------------------------------------- construction
    @classmethod
    def from_gguf(cls, meta: dict) -> "Tokenizer":
        model = meta.get("tokenizer.ggml.model")
        tokens = meta.get("tokenizer.ggml.tokens")
        if model not in ("gpt2", "llama") or not isinstance(tokens, list) or not tokens:
            raise UnsupportedModel(f"The built-in engine cannot read this model's tokenizer ({model or 'none found'}). "
                                   "It supports byte-level BPE (Qwen, Llama 3, SmolLM) and SentencePiece (Llama 2, Mistral, TinyLlama).")
        types = meta.get("tokenizer.ggml.token_type") or [NORMAL] * len(tokens)
        if len(types) != len(tokens):
            raise UnsupportedModel("The model's tokenizer data is inconsistent (token and token type counts differ).")

        def token_id(key):
            v = meta.get(key)
            return v if isinstance(v, int) and 0 <= v < len(tokens) else None

        eos_id = token_id("tokenizer.ggml.eos_token_id")
        eos = {i for i in (eos_id, token_id("tokenizer.ggml.eot_token_id"), token_id("tokenizer.ggml.eom_token_id")) if i is not None}
        eos |= {i for i, t in enumerate(tokens) if t in _END_OF_TURN and types[i] in (CONTROL, USER_DEFINED)}
        common = dict(tokens=tokens, types=types, bos_id=token_id("tokenizer.ggml.bos_token_id"), eos_id=eos_id, eos_ids=eos,
                      chat_template=meta.get("tokenizer.chat_template") if isinstance(meta.get("tokenizer.chat_template"), str) else None,
                      unk_id=token_id("tokenizer.ggml.unknown_token_id"))
        if model == "gpt2":
            return _ByteLevelBPE(add_bos=bool(meta.get("tokenizer.ggml.add_bos_token", False)),
                                 merges=meta.get("tokenizer.ggml.merges"), pre=meta.get("tokenizer.ggml.pre") or "default", **common)
        return _SentencePiece(add_bos=bool(meta.get("tokenizer.ggml.add_bos_token", True)),
                              scores=meta.get("tokenizer.ggml.scores"),
                              add_space_prefix=bool(meta.get("tokenizer.ggml.add_space_prefix", True)), **common)

    # ----------------------------------------------------------------------------------------------------- encoding
    def encode(self, text: str, add_bos: bool = False, parse_special: bool = True) -> list[int]:
        """Text -> token ids. With `parse_special`, special tokens written out in the text ("<|im_start|>") become their ids.

        A BOS token is never doubled: `add_bos=True` is safe for chat prompts whose template already starts with it.
        """
        ids: list[int] = []
        after_special = True
        for is_token, payload in self._fragments(text, parse_special):
            if is_token:
                ids.append(payload)
            else:
                ids.extend(self._encode_text(payload, after_special))
            after_special = is_token
        if add_bos and self.bos_id is not None and (not ids or ids[0] != self.bos_id):
            ids.insert(0, self.bos_id)
        return ids

    def _partition_pattern(self, parse_special: bool):
        if parse_special not in self._partitions:
            kinds = (CONTROL, USER_DEFINED) if parse_special else (USER_DEFINED,)
            specials = sorted({t for t, k in zip(self.tokens, self.types) if k in kinds and t}, key=len, reverse=True)
            self._partitions[parse_special] = re.compile("|".join(map(re.escape, specials))) if specials else None
        return self._partitions[parse_special]

    def _fragments(self, text: str, parse_special: bool):
        """Split `text` into (False, plain text) and (True, token id) fragments."""
        rx = self._partition_pattern(parse_special)
        pos = 0
        for m in (rx.finditer(text) if rx else ()):
            if m.start() > pos:
                yield False, text[pos:m.start()]
            yield True, self._token_ids[m.group()]
            pos = m.end()
        if pos < len(text):
            yield False, text[pos:]

    def _encode_text(self, text: str, after_special: bool) -> list[int]:
        raise NotImplementedError

    # ----------------------------------------------------------------------------------------------------- decoding
    def piece(self, token_id: int, special: bool = False) -> bytes:
        """The bytes a token contributes to the output. Control tokens are invisible unless `special`."""
        if not 0 <= token_id < self.vocab_size:
            raise ValueError(f"token id {token_id} is outside the vocabulary (0..{self.vocab_size - 1})")
        cache = self._special_pieces if special else self._pieces
        piece = cache[token_id]
        if piece is None:
            kind = self.types[token_id]
            if kind in (CONTROL, UNKNOWN, UNUSED) and not special:
                piece = b""
            else:
                piece = self._token_bytes(token_id, kind)
            cache[token_id] = piece
        return piece

    def _token_bytes(self, token_id: int, kind: int) -> bytes:
        raise NotImplementedError

    def decode(self, ids: list[int], special: bool = False) -> str:
        """Token ids -> text. Control tokens such as <|im_end|> are dropped unless `special`."""
        dec = self.stream_decoder(special)
        return dec.push_all(ids) + dec.flush()

    def stream_decoder(self, special: bool = False) -> "StreamDecoder":
        return StreamDecoder(self, special, self._strip_first_space)

    # ----------------------------------------------------------------------------------------------- chat templates
    def apply_chat_template(self, messages: list[dict], add_generation_prompt: bool = True) -> str:
        """Render a conversation in the model's own prompt format (decided from its GGUF chat template, never executed).

        The result includes the BOS text for families whose template emits it; encode it with `parse_special=True`.
        """
        turns = _messages(messages)
        return getattr(self, "_render_" + self.chat_format)(turns, add_generation_prompt)

    def _bos_text(self) -> str:
        return self.tokens[self.bos_id] if self.bos_id is not None and "bos_token" in self.chat_template else ""

    def _eos_text(self) -> str:
        return self.tokens[self.eos_id] if self.eos_id is not None else ""

    def _render_chatml(self, turns, add_generation_prompt: bool) -> str:
        t = self.chat_template
        out = []
        default = re.search(r"<\|im_start\|>system\\n([^'\"{}]+)<\|im_end\|>\\n", t)
        if turns[0][0] != "system" and default:
            out.append(f"<|im_start|>system\n{default.group(1)}<|im_end|>\n")
        for role, text in turns:
            if role == "assistant" and "</think>" in t:
                text = text.split("</think>")[-1].lstrip("\n")  # Qwen3 templates drop earlier reasoning
            out.append(f"<|im_start|>{role}\n{text}<|im_end|>\n")
        if add_generation_prompt:
            thinking = re.search(r"<\|im_start\|>assistant\\n<think>\\n'", t)
            out.append("<|im_start|>assistant\n" + ("<think>\n" if thinking else ""))
        return "".join(out)

    def _render_llama3(self, turns, add_generation_prompt: bool) -> str:
        out = [self._bos_text()]
        if "Cutting Knowledge Date" in self.chat_template:  # Llama 3.1 / 3.2 always open with a dated system block
            system = turns[0][1].strip() if turns[0][0] == "system" else ""
            date = _today() if "strftime_now" in self.chat_template else "26 Jul 2024"
            out.append("<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n"
                       f"Today Date: {date}\n\n{system}<|eot_id|>")
            turns = turns[1:] if turns[0][0] == "system" else turns
        for role, text in turns:
            out.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{text.strip()}<|eot_id|>")
        if add_generation_prompt:
            out.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(out)

    def _render_gemma(self, turns, add_generation_prompt: bool) -> str:
        out = [self._bos_text()]
        prefix = ""
        if turns[0][0] == "system":  # Gemma has no system role: its text leads the first user turn
            prefix, turns = turns[0][1] + "\n\n", turns[1:]
        for role, text in turns:
            out.append(f"<start_of_turn>{'model' if role == 'assistant' else 'user'}\n{prefix}{text.strip()}<end_of_turn>\n")
            prefix = ""
        if add_generation_prompt:
            out.append("<start_of_turn>model\n")
        return "".join(out)

    def _render_phi3(self, turns, add_generation_prompt: bool) -> str:
        out = [self._bos_text()] + [f"<|{role}|>\n{text}<|end|>\n" for role, text in turns]
        if add_generation_prompt:
            out.append("<|assistant|>\n")
        return "".join(out)

    def _render_zephyr(self, turns, add_generation_prompt: bool) -> str:
        eos = self._eos_text()
        out = [self._bos_text()] + [f"<|{role}|>\n{text}{eos}\n" for role, text in turns]
        if add_generation_prompt:
            out.append("<|assistant|>\n")
        return "".join(out)

    def _render_mistral(self, turns, add_generation_prompt: bool) -> str:
        spaced = " [/INST]" in self.chat_template  # v0.1 / v0.2 write "[INST] x [/INST]"; v0.3 and later "[INST] x[/INST]"
        system = ""
        if turns[0][0] == "system":
            system, turns = turns[0][1], turns[1:]
        users = [i for i, (role, _) in enumerate(turns) if role == "user"]
        if system and not users:
            turns, system, users = [("user", system)] + turns, "", [0]
        carrier = users[0] if spaced else users[-1] if users else None  # the user turn that carries the system text
        out = [self._bos_text()]
        for i, (role, text) in enumerate(turns):
            if role == "assistant":
                out.append(text + self._eos_text())
            else:
                prefix = f"{system}\n\n" if system and i == carrier else ""
                out.append(f"[INST] {prefix}{text}{' ' if spaced else ''}[/INST]")
        return "".join(out)

    def _render_llama2(self, turns, add_generation_prompt: bool) -> str:
        system = turns[0][1] if turns[0][0] == "system" else ""
        turns = turns[1:] if turns[0][0] == "system" else turns
        out = []
        for i, (role, text) in enumerate(turns):
            text = text.strip()
            if role == "assistant":
                out.append(f" {text} {self._eos_text()}")
            else:
                if system and i == 0:
                    text = f"<<SYS>>\n{system}\n<</SYS>>\n\n{text}"
                out.append(f"{self._bos_text()}[INST] {text} [/INST]")
        return "".join(out)


class StreamDecoder:
    """Turns token ids into text one at a time, holding back bytes of a character that is still incomplete."""

    def __init__(self, tokenizer: Tokenizer, special: bool = False, strip_first_space: bool = False):
        self._tok, self._special, self._strip = tokenizer, special, strip_first_space
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def push(self, token_id: int) -> str:
        return self._emit(self._decoder.decode(self._tok.piece(token_id, self._special)))

    def push_all(self, ids) -> str:
        return "".join(self.push(i) for i in ids)

    def flush(self) -> str:
        return self._emit(self._decoder.decode(b"", final=True))

    def _emit(self, text: str) -> str:
        if self._strip and text:
            self._strip = False
            return text[1:] if text[0] == " " else text  # SentencePiece's dummy prefix space
        return text


# ------------------------------------------------------------------------------------------------------ byte-level BPE
class _ByteLevelBPE(Tokenizer):
    def __init__(self, *, merges, pre: str, **kw):
        super().__init__(**kw)
        if pre not in PRE_TOKENIZERS:
            raise UnsupportedModel(f"The built-in engine does not know this model's pre-tokenizer ({pre!r}); "
                                   f"it supports {', '.join(sorted(PRE_TOKENIZERS))}. Use the llama.cpp engine.")
        if not isinstance(merges, list) or not merges:
            raise UnsupportedModel("This model's tokenizer has no BPE merges.")
        self._regexes = _compiled(tuple(PRE_TOKENIZERS[pre]))
        self._to_char, self._to_byte = _byte_unicode()
        self._ranks: dict[tuple[str, str], int] = {}
        for rank, merge in enumerate(merges):
            a, _, b = merge.partition(" ")
            if b and a + b in self._token_ids:
                self._ranks.setdefault((a, b), rank)
        self._cache: dict[str, tuple[int, ...]] = {}

    def _encode_text(self, text: str, after_special: bool) -> list[int]:
        ids: list[int] = []
        for piece in _split(text, self._regexes):
            word = piece.encode("utf-8").decode("latin-1").translate(self._to_char)
            cached = self._cache.get(word)
            if cached is None:
                if len(self._cache) >= _MAX_CACHE:
                    self._cache.clear()
                cached = self._cache[word] = tuple(self._merge(word))
            ids.extend(cached)
        return ids

    def _merge(self, word: str) -> list[int]:
        """Greedy BPE: repeatedly merge the adjacent pair with the lowest merge rank (leftmost first), in O(n log n)."""
        syms: list[str | None] = list(word)
        n = len(syms)
        nxt, prv = list(range(1, n + 1)), list(range(-1, n - 1))
        nxt[-1] = -1
        ranks, heap = self._ranks, []

        def offer(i: int) -> None:
            j = nxt[i]
            if j >= 0:
                rank = ranks.get((syms[i], syms[j]))
                if rank is not None:
                    heapq.heappush(heap, (rank, i, syms[i], syms[j]))

        for i in range(n - 1):
            offer(i)
        while heap:
            _, i, a, b = heapq.heappop(heap)
            j = nxt[i]
            if syms[i] != a or j < 0 or syms[j] != b:
                continue  # stale entry: one side was merged since it was queued
            syms[i], syms[j] = a + b, None
            nxt[i] = nxt[j]
            if nxt[j] >= 0:
                prv[nxt[j]] = i
            if prv[i] >= 0:
                offer(prv[i])
            offer(i)
        ids, i = [], 0
        while i >= 0:
            token = self._token_ids.get(syms[i])
            if token is not None:
                ids.append(token)
            elif self.unk_id is not None:
                ids.append(self.unk_id)
            i = nxt[i]
        return ids

    def _token_bytes(self, token_id: int, kind: int) -> bytes:
        text = self.tokens[token_id]
        if kind == BYTE and re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", text):
            return bytes([int(text[3:5], 16)])
        if kind in (NORMAL, BYTE):
            return b"".join(bytes([self._to_byte[c]]) if c in self._to_byte else c.encode("utf-8") for c in text)
        return text.encode("utf-8")


# ------------------------------------------------------------------------------------------------------- SentencePiece
class _SentencePiece(Tokenizer):
    """llama.cpp's SentencePiece tokenizer: split into characters, repeatedly merge the pair whose merged piece scores highest,
    and write characters that are not in the vocabulary as <0xXX> byte tokens."""

    def __init__(self, *, scores, add_space_prefix: bool, **kw):
        super().__init__(**kw)
        if not isinstance(scores, list) or len(scores) != self.vocab_size:
            raise UnsupportedModel("This model's SentencePiece tokenizer has no usable scores.")
        self._scores = scores
        self.add_space_prefix = add_space_prefix
        self._strip_first_space = add_space_prefix
        self._merge_ids = {t: i for i, t in enumerate(self.tokens) if self.types[i] == NORMAL}
        self._byte_ids = [self._token_ids.get(f"<0x{b:02X}>") for b in range(256)]

    def _encode_text(self, text: str, after_special: bool) -> list[int]:
        if self.add_space_prefix and after_special:
            text = " " + text
        syms: list[str | None] = list(text.replace(" ", SEPARATOR))
        n = len(syms)
        if n == 0:
            return []
        nxt, prv = list(range(1, n + 1)), list(range(-1, n - 1))
        nxt[-1] = -1
        ids, scores, heap = self._merge_ids, self._scores, []

        def offer(i: int) -> None:
            j = nxt[i]
            if j >= 0:
                token = ids.get(syms[i] + syms[j])
                if token is not None:
                    heapq.heappush(heap, (-scores[token], i, syms[i], syms[j]))

        for i in range(n - 1):
            offer(i)
        while heap:
            _, i, a, b = heapq.heappop(heap)
            j = nxt[i]
            if syms[i] != a or j < 0 or syms[j] != b:
                continue
            syms[i], syms[j] = a + b, None
            nxt[i] = nxt[j]
            if nxt[j] >= 0:
                prv[nxt[j]] = i
            if prv[i] >= 0:
                offer(prv[i])
            offer(i)
        out, i = [], 0
        while i >= 0:
            token = ids.get(syms[i])
            if token is not None:
                out.append(token)
            else:
                for byte in syms[i].encode("utf-8"):
                    b = self._byte_ids[byte]
                    if b is not None:
                        out.append(b)
                    elif self.unk_id is not None:
                        out.append(self.unk_id)
            i = nxt[i]
        return out

    def _token_bytes(self, token_id: int, kind: int) -> bytes:
        text = self.tokens[token_id]
        if kind == BYTE and re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", text):
            return bytes([int(text[3:5], 16)])
        return text.replace(SEPARATOR, " ").encode("utf-8")
