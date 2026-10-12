"""Training data: rated, corrected and big-model answers turned into reviewable examples and a training bundle.

`DatasetStore.refresh()` derives two kinds of example from the recorded exchanges: "sft" (a prompt and the best
answer to it) and "pref" (the same prompt with a better and a worse answer). Nothing is used for training until
the user approves it. `export_zip()` then packs the approved examples, optionally redacted, together with a
LoRA training script and a plain-language guide for a rented GPU.
"""

from __future__ import annotations

import datetime
import io
import json
import math
import re
import time
import zipfile
from collections import Counter
from collections.abc import Iterator
from typing import TYPE_CHECKING

from . import __version__
from .db import Database

if TYPE_CHECKING:
    from .feedback import ExchangeStore

KINDS = ("sft", "pref")
STATUSES = ("pending", "approved", "rejected")
SOURCES = ("corrected", "teacher", "liked")
MIN_CHOSEN = 8                        # shorter answers carry nothing worth training on
MAX_TEXT = 200_000                    # an edited answer
MAX_INT = 2**31
MAX_LIMIT = 200                       # rows per list() page
MAX_IDS = 5000                        # items changed by one set_status() call
SCAN_PAGE = 50                        # exchanges examined per query in refresh()
SQL_CHUNK = 400                       # ids per IN (...) clause

_ROLES = ("system", "user", "assistant")
_ITEM_COLUMNS = "id, exchange_id, kind, messages_json, chosen, rejected, source, status, edited, created"
_CANDIDATES = """SELECT id, answer, rating, correction, teacher_answer, messages_json FROM exchanges
                 WHERE id > ? AND (correction != '' OR teacher_answer != '' OR rating = 1)
                 ORDER BY id LIMIT ?"""


# ------------------------------------------------------------------ text helpers
def _dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _loads_list(raw) -> list:
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return value if isinstance(value, list) else []


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def _like(text: str) -> str:
    return "%" + text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def _as_id(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 0 < value < MAX_INT else None


def _marks(values) -> str:
    """SQL placeholders for an IN (...) clause; only these are ever formatted into a statement."""
    return ",".join("?" * len(values))


def _chunks(values: list, size: int = SQL_CHUNK) -> Iterator[list]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _clamp(value, low: int, high: int) -> int:
    try:
        return max(low, min(int(value), high))
    except (TypeError, ValueError):
        return low


def _text_arg(value, what: str) -> str:
    """User-supplied training text: a non-empty string without NULs or lone surrogates, within MAX_TEXT."""
    if not isinstance(value, str):
        raise ValueError(f"{what} must be text")
    text = value.replace("\x00", "").encode("utf-8", "replace").decode("utf-8").strip()
    if not text:
        raise ValueError(f"{what} can't be empty")
    if len(text) > MAX_TEXT:
        raise ValueError(f"{what} is too long (at most {MAX_TEXT:,} characters)")
    return text


def _content_text(content) -> str:
    """Plain text of a message's content, which may be a string or a list of parts."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type", "text") == "text" and isinstance(part.get("text"), str):
            parts.append(part["text"])
    return "\n".join(parts)


def _prompt_turns(raw: str) -> list[dict] | None:
    """The chat turns of a stored conversation as plain {role, content} pairs.

    None when the answer cannot be trained on without more context: the conversation does not end with a
    user message (the answer followed tool output the training data would not contain). Tool turns that came
    earlier in the conversation are dropped.
    """
    messages = _loads_list(raw)
    last = messages[-1] if messages else None
    if not isinstance(last, dict) or last.get("role") != "user" or not _content_text(last.get("content")).strip():
        return None
    turns = []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in _ROLES or m.get("tool_calls"):
            continue
        text = _content_text(m.get("content")).strip()
        if text:
            turns.append({"role": m["role"], "content": text})
    return turns


def _stored_turns(raw: str) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in _loads_list(raw)
            if isinstance(m, dict) and isinstance(m.get("role"), str) and isinstance(m.get("content"), str)]


def _item(row: dict) -> dict:
    row["messages"] = _loads_list(row.pop("messages_json"))
    row["edited"] = bool(row["edited"])
    return row


# ------------------------------------------------------------------ deriving examples
def _best_answer(row: dict) -> tuple[str, str]:
    """The answer to learn from and where it came from, or ("", "") when there is none."""
    correction, teacher, answer = row["correction"].strip(), row["teacher_answer"].strip(), row["answer"].strip()
    if correction:
        return correction, "corrected"
    if teacher:
        return teacher, "teacher"
    if row["rating"] == 1 and answer:
        return answer, "liked"
    return "", ""


def _derive(row: dict) -> list[dict]:
    """The dataset items one exchange row yields (none, an sft item, or an sft and a pref item)."""
    chosen, source = _best_answer(row)
    prompt = _prompt_turns(row["messages_json"]) if len(chosen) >= MIN_CHOSEN else None
    if prompt is None:
        return []
    base = {"messages_json": _dumps(prompt), "chosen": chosen, "source": source}
    items = [{**base, "kind": "sft", "rejected": ""}]
    original = row["answer"].strip()
    if source != "liked" and original and _norm(original) != _norm(chosen):
        items.append({**base, "kind": "pref", "rejected": original})
    return items


_FIELDS = ("messages_json", "chosen", "rejected", "source")


class DatasetStore:
    def __init__(self, db: Database, exchanges: ExchangeStore):
        self.db = db
        self.exchanges = exchanges

    # ------------------------------------------------------------------ building
    def refresh(self) -> dict:
        """Derive items from the recorded exchanges; safe to call as often as you like.

        New items start as "pending". Pending items nobody has edited follow their exchange (a changed answer
        updates them, a vanished one removes them); approved, rejected and edited items are never touched.
        An item the user deleted comes back here while its exchange still qualifies - reject it instead.
        """
        wanted: set[tuple[int, str]] = set()
        added, last = 0, 0
        with self.db.transaction():
            while True:
                rows = self.db.query(_CANDIDATES, (last, SCAN_PAGE))
                if not rows:
                    break
                last = rows[-1]["id"]
                existing = self._existing([r["id"] for r in rows])
                for row in rows:
                    for fields in _derive(row):
                        key = (row["id"], fields["kind"])
                        wanted.add(key)
                        added += self._store(key, fields, existing.get(key))
            self._prune(wanted)
            total = self.db.scalar("SELECT count(*) FROM dataset_items")
        return {"added": added, "total": total}

    def _existing(self, exchange_ids: list[int]) -> dict:
        rows = self.db.query(f"""SELECT id, exchange_id, kind, messages_json, chosen, rejected, source, status, edited
                                 FROM dataset_items WHERE exchange_id IN ({_marks(exchange_ids)})""", exchange_ids)
        return {(r["exchange_id"], r["kind"]): r for r in rows}

    def _store(self, key: tuple[int, str], fields: dict, existing: dict | None) -> int:
        """Insert or refresh one item; returns 1 when a new item was added."""
        if existing is None:
            self.db.execute("""INSERT INTO dataset_items(exchange_id, kind, messages_json, chosen, rejected, source,
                                                         created) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            (key[0], key[1], fields["messages_json"], fields["chosen"], fields["rejected"],
                             fields["source"], time.time()))
            return 1
        untouched = existing["status"] == "pending" and not existing["edited"]
        if untouched and any(existing[f] != fields[f] for f in _FIELDS):
            self.db.execute("UPDATE dataset_items SET messages_json = ?, chosen = ?, rejected = ?, source = ? "
                            "WHERE id = ?", (*(fields[f] for f in _FIELDS), existing["id"]))
        return 0

    def _prune(self, wanted: set[tuple[int, str]]) -> None:
        stale = [r["id"] for r in self.db.query(
            "SELECT id, exchange_id, kind FROM dataset_items WHERE status = 'pending' AND edited = 0")
            if (r["exchange_id"], r["kind"]) not in wanted]
        for part in _chunks(stale):
            self.db.execute(f"DELETE FROM dataset_items WHERE id IN ({_marks(part)})", part)

    # ------------------------------------------------------------------ reviewing
    def list(self, status="", kind="", q="", limit=50, offset=0) -> dict:
        where, params = [], []
        for column, value in (("status", status), ("kind", kind)):
            if value:
                where.append(f"{column} = ?")
                params.append(str(value))
        q = str(q or "").replace("\x00", "").strip()[:200]
        if q:
            where.append("(" + " OR ".join(f"{c} LIKE ? ESCAPE '\\'" for c in ("chosen", "rejected", "messages_json"))
                         + ")")
            params += [_like(q)] * 3
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        total = self.db.scalar(f"SELECT count(*) FROM dataset_items{clause}", params)
        rows = self.db.query(f"SELECT {_ITEM_COLUMNS} FROM dataset_items{clause} ORDER BY id DESC LIMIT ? OFFSET ?",
                             [*params, _clamp(limit, 1, MAX_LIMIT), _clamp(offset, 0, MAX_INT)])
        return {"items": [_item(r) for r in rows], "total": total}

    def get(self, id) -> dict | None:
        id = _as_id(id)
        row = None if id is None else self.db.one(f"SELECT {_ITEM_COLUMNS} FROM dataset_items WHERE id = ?", (id,))
        return _item(row) if row else None

    def set_status(self, ids: list[int], status) -> int:
        if status not in STATUSES:
            raise ValueError("status must be pending, approved or rejected")
        if not isinstance(ids, (list, tuple)) or len(ids) > MAX_IDS:
            raise ValueError(f"ids must be a list of at most {MAX_IDS} item ids")
        wanted = sorted({i for i in map(_as_id, ids) if i is not None})
        updated = 0
        with self.db.transaction():
            for part in _chunks(wanted):
                updated += self.db.execute(f"UPDATE dataset_items SET status = ? WHERE id IN ({_marks(part)})",
                                           [status, *part]).rowcount
        return updated

    def edit(self, id, chosen, rejected=None) -> dict | None:
        """Replace the answers of an item and mark it edited. `rejected` only applies to "pref" items."""
        item = self.get(id)
        if item is None:
            return None
        chosen = _text_arg(chosen, "The chosen answer")
        if item["kind"] == "pref":
            worse = item["rejected"] if rejected is None else _text_arg(rejected, "The rejected answer")
            if _norm(worse) == _norm(chosen):
                raise ValueError("The chosen and rejected answers are the same")
        else:
            worse = ""
        self.db.execute("UPDATE dataset_items SET chosen = ?, rejected = ?, edited = 1 WHERE id = ?",
                        (chosen, worse, item["id"]))
        return self.get(item["id"])

    def delete(self, id) -> bool:
        id = _as_id(id)
        return id is not None and self.db.execute("DELETE FROM dataset_items WHERE id = ?", (id,)).rowcount > 0

    def stats(self) -> dict:
        return self.db.one("""SELECT coalesce(sum(status = 'pending'), 0) AS pending,
                                     coalesce(sum(status = 'approved'), 0) AS approved,
                                     coalesce(sum(status = 'rejected'), 0) AS rejected,
                                     coalesce(sum(kind = 'sft'), 0) AS sft, coalesce(sum(kind = 'pref'), 0) AS pref,
                                     count(*) AS total FROM dataset_items""")

    # ------------------------------------------------------------------ exporting
    def export_zip(self, only_approved=True, redact=True, include_reasoning=False, include_script=True) -> bytes:
        """A zip with sft.jsonl, pref.jsonl, manifest.json, README_TRAINING.md and (optionally) train_lora.py.

        Rejected items are never exported; without `only_approved` the pending ones are included too.
        `include_reasoning` adds the model's chain of thought to answers that are its own unedited answer.
        """
        statuses = ("approved",) if only_approved else ("approved", "pending")
        rows = self.db.query(f"SELECT {_ITEM_COLUMNS} FROM dataset_items WHERE status IN ({_marks(statuses)}) "
                             f"ORDER BY id", statuses)
        scrub = _redact if redact else str
        lines: dict[str, list[dict]] = {"sft": [], "pref": []}
        sources = {kind: dict.fromkeys(SOURCES, 0) for kind in KINDS}
        for row in rows:
            thoughts = self._thoughts(row) if include_reasoning else {}
            example = _example(row, scrub, thoughts)
            if example is not None:
                lines[row["kind"]].append(example)
                sources[row["kind"]][row["source"]] = sources[row["kind"]].get(row["source"], 0) + 1
        if not lines["sft"] and not lines["pref"]:
            raise ValueError("No approved training examples yet - approve some in Learning > Training data first."
                             if only_approved else "There are no training examples to export yet.")
        now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
        manifest = {"created": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "duckstoken_version": __version__,
                    "redacted": bool(redact), "only_approved": bool(only_approved),
                    "include_reasoning": bool(include_reasoning),
                    "counts": {kind: len(lines[kind]) for kind in KINDS}, "sources": sources}
        files = [("sft.jsonl", _jsonl(lines["sft"])), ("pref.jsonl", _jsonl(lines["pref"])),
                 ("manifest.json", json.dumps(manifest, indent=2) + "\n"), ("README_TRAINING.md", training_readme())]
        if include_script:
            files.append(("train_lora.py", train_script()))
        return _zip(files, now)

    def _thoughts(self, row: dict) -> dict[str, str]:
        """{normalized answer: reasoning} for the exchange's original answer, when it recorded any reasoning."""
        exchange = self.exchanges.get(row["exchange_id"])
        if not exchange or not exchange["reasoning"].strip() or not exchange["answer"].strip():
            return {}
        return {_norm(exchange["answer"]): exchange["reasoning"].strip()}


def _example(row: dict, scrub, thoughts: dict[str, str]) -> dict | None:
    """One line of sft.jsonl or pref.jsonl for a dataset item, or None when the item is unusable."""
    prompt = [{"role": m["role"], "content": scrub(m["content"])} for m in _stored_turns(row["messages_json"])]

    def reply(text: str) -> list[dict]:
        message = {"role": "assistant", "content": scrub(text)}
        if _norm(text) in thoughts:
            message["thinking"] = scrub(thoughts[_norm(text)])
        return [message]

    if not prompt:
        return None
    if row["kind"] == "sft":
        return {"messages": [*prompt, *reply(row["chosen"])]}
    if row["kind"] == "pref" and row["rejected"]:
        return {"prompt": prompt, "chosen": reply(row["chosen"]), "rejected": reply(row["rejected"])}
    return None


def _jsonl(objects: list[dict]) -> str:
    """One JSON object per line. The Unicode line separators JSON allows raw are escaped so any splitter works."""
    lines = (json.dumps(o, ensure_ascii=False) for o in objects)
    return "".join(line.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029").replace("\x85", "\\u0085") + "\n"
                   for line in lines)


def _zip(files: list[tuple[str, str]], when: datetime.datetime) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, text in files:
            info = zipfile.ZipInfo(name, date_time=when.timetuple()[:6])
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, text.encode("utf-8", "replace"))
    return buffer.getvalue()


# ------------------------------------------------------------------ redaction
def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


def _card(m: re.Match) -> str:
    digits = _digits(m.group())
    return "<number>" if 13 <= len(digits) <= 19 and digits[0] in "3456" and _luhn(digits) else m.group()


def _phone(m: re.Match) -> str:
    return "<phone>" if 9 <= len(_digits(m.group())) <= 15 else m.group()


def _entropy(text: str) -> float:
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in Counter(text).values())


def _random_blob(m: re.Match) -> str:
    """Long runs that look like generated secrets, not like words, CamelCase identifiers or paths."""
    token = m.group()
    runs = [len(w) for w in re.findall(r"[a-z]+", token)]  # lowercase stretches: long and regular in real names
    symbols = sum(not c.isalnum() for c in token)
    words = [w for w in re.split(r"[^A-Za-z0-9]+", token) if len(w) >= 3 and w.isalpha() and w.islower()]
    capitals = [i for i, c in enumerate(token[:-1]) if c.isupper()]
    camel_case = (len(capitals) >= 4 and sum(token[i + 1].islower() for i in capitals) / len(capitals) > 0.7
                  and sum(runs) / max(len(runs), 1) >= 2.5)
    mixed = any(c.islower() for c in token) and any(c.isupper() for c in token) and any(c.isdigit() for c in token)
    random_like = (mixed and not camel_case and len(words) < 2 and max(runs, default=0) < 10
                   and symbols <= len(token) // 8 and _entropy(token) >= 4.0)
    return "<secret>" if random_like else token


def _hex_blob(m: re.Match) -> str:
    token = m.group()
    return "<secret>" if re.search(r"\d", token) and re.search(r"[a-fA-F]", token) else token


_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_TOKEN_CHARS = r"A-Za-z0-9+/_\-"
_EMAIL = r"(?<![\w.+%\-])[\w.+%\-]{1,64}@(?:[^\W_][\w\-]{0,62}\.){1,8}[^\W\d_]{2,24}(?!\w)"
_KEYS = "|".join((
    r"\bsk-(?:ant-)?[A-Za-z0-9_\-]{20,}",                              # OpenAI, Anthropic
    r"\bgh[pousr]_[A-Za-z0-9]{30,}", r"\bgithub_pat_[A-Za-z0-9_]{20,}",  # GitHub
    r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",                                  # AWS access key ids
    r"\bxox[abprs]-[A-Za-z0-9\-]{10,}",                                # Slack
    r"\bAIza[0-9A-Za-z_\-]{35}",                                       # Google API keys
    r"\bhf_[A-Za-z0-9]{30,}",                                          # Hugging Face
    r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}",                         # Stripe
    r"\beyJ[A-Za-z0-9_\-]{5,}\.[A-Za-z0-9_\-]{5,}\.[A-Za-z0-9_\-]*",  # JWT
))
_PRIVATE_KEY = r"-----BEGIN [A-Z ]{0,30}PRIVATE KEY-----[\s\S]{0,8000}?-----END [A-Z ]{0,30}PRIVATE KEY-----"
_PHONE = r"""(?<![\w.\-/])(?:
      \+[1-9]\d{0,2}[\s.\-]?(?:\(\d{1,4}\)[\s.\-]?)?\d{1,4}(?:[\s.\-]?\d{2,4}){1,4}   # +44 20 7946 0958
    | (?:\(\d{3}\)\s?|\d{3}[\s.\-])\d{3}[\s.\-]\d{4}                                  # (555) 123-4567
    | 0\d{1,4}(?:[\s.\-]\d{2,4}){2,4}                                                 # 06 12 34 56 78
    )(?!\w|[.\-/]\d)"""
_WINDOWS_HOME = (r"(?i)\b([A-Z]:[\\/]Users[\\/])(?!(?:Public|Default|All Users)[\\/])(?!\s)"
                 r"[^\\/:*?\"<>|\r\n]{1,64}(?=[\\/])")
_UNIX_HOME = r"(?<![\w.\-:~])((?:/home|/Users)/)[^/\s<>:*?\"|]{1,64}(?=/)"


def _ip(m: re.Match) -> str:
    return m.group() if m.group() in ("127.0.0.1", "0.0.0.0") else "<ip>"


# Applied in order; each entry is (pattern, replacement string or function).
_REDACTIONS: list[tuple[re.Pattern, object]] = [
    (re.compile(_PRIVATE_KEY), "<secret>"),
    (re.compile(_EMAIL), "<email>"),
    (re.compile(_KEYS), "<secret>"),
    (re.compile(r"\b([Bb]earer\s+)(?=[A-Za-z0-9._~+/\-]*\d)[A-Za-z0-9._~+/\-]{16,}=*"), r"\1<secret>"),
    (re.compile(r"(?<![\w.\-])\d(?:[ \-]?\d){12,18}(?!\w)"), _card),
    (re.compile(_PHONE, re.X), _phone),
    (re.compile(rf"(?<![\w.])(?:{_OCTET}\.){{3}}{_OCTET}(?![\w]|\.\d)"), _ip),
    (re.compile(r"(?<![0-9A-Za-z])[0-9a-fA-F]{32,}(?![0-9A-Za-z])"), _hex_blob),
    (re.compile(rf"(?<![{_TOKEN_CHARS}])[{_TOKEN_CHARS}]{{32,}}={{0,2}}(?![{_TOKEN_CHARS}])"), _random_blob),
    (re.compile(_WINDOWS_HOME), r"\1<user>"),
    (re.compile(_UNIX_HOME), r"\1<user>"),
]


def redact(text: str) -> str:
    """`text` with emails, phone numbers, API keys and tokens, IPv4 addresses, card numbers and user names in
    home folders replaced by placeholders. Best effort: names, addresses and passwords in plain prose stay."""
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


_redact = redact  # export_zip() has a parameter called `redact`


# ------------------------------------------------------------------ the training bundle
def train_script() -> str:
    """The text of train_lora.py: an UNRUN template for a LoRA fine-tune of gpt-oss-20b."""
    return _TRAIN_SCRIPT


def training_readme() -> str:
    """The text of README_TRAINING.md."""
    return _README


_TRAIN_SCRIPT = r'''#!/usr/bin/env python3
"""LoRA fine-tune of openai/gpt-oss-20b on your approved Duckstoken training examples.

======================================================================================================
THIS FILE IS A TEMPLATE. It has NOT been run inside the Duckstoken repository, and the libraries it
uses (transformers, peft, trl) change their APIs often. It follows the approach of OpenAI's cookbook
article "How to fine-tune gpt-oss" (transformers >= 4.56, peft >= 0.17, trl >= 0.21). If something
fails, compare this file with that article and with the current TRL "SFTTrainer" documentation.
======================================================================================================

What it does
    Reads sft.jsonl (one {"messages": [...]} chat per line), trains a small LoRA adapter on top of
    gpt-oss-20b and saves the adapter. The base model's own weights are not changed.

What you need
    * An NVIDIA GPU with roughly 48-80 GB of memory. gpt-oss ships its weights in 4-bit MXFP4, but
      training expands them to bf16 (about 40 GB) plus activations, so an 80 GB A100/H100 is comfortable
      and a 48 GB card only just fits with the small batch settings below. A 24 GB gaming card will not do.
    * About 60 GB of free disk space (the base model download and caches).
    * Python 3.10+ and, ideally, a fresh virtual environment:
          pip install -U torch "transformers>=4.56" "peft>=0.17" "trl>=0.21" datasets accelerate
      (most GPU rental images already ship a matching PyTorch/CUDA build)

How to run
    python train_lora.py        # in the folder that contains sft.jsonl

After training: turn the adapter into a GGUF file, which is the only adapter format llama.cpp loads
    git clone https://github.com/ggml-org/llama.cpp
    pip install -r llama.cpp/requirements.txt
    python llama.cpp/convert_lora_to_gguf.py --base-model-id openai/gpt-oss-20b --outtype f16 --outfile duckstoken-lora.gguf duckstoken-lora
    (Whether your llama.cpp version can convert gpt-oss adapters is untested here - use a recent one and
     check its issue tracker if the converter complains.)

Use it in Duckstoken
    1. Copy duckstoken-lora.gguf to the PC that runs Duckstoken.
    2. Settings -> "LoRA adapter (.gguf)": pick that file. Leave "LoRA strength" at 1.0, or lower it
       (for example 0.6) to blend the fine-tune with the original behaviour. Restart the model.
    3. The adapter only fits openai/gpt-oss-20b. It cannot be loaded on gpt-oss-120b or any other model,
       because a LoRA is tied to the exact layer shapes of the model it was trained on.
"""

import json
import sys
from pathlib import Path

MODEL_ID = "openai/gpt-oss-20b"
HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "sft.jsonl"
OUTPUT_DIR = HERE / "duckstoken-lora"

# ---- settings you may want to change -----------------------------------------------------------
EPOCHS = 3               # 2-3 suits a few hundred examples; use fewer for thousands
LEARNING_RATE = 2e-4     # the usual LoRA value
LORA_RANK = 16           # adapter capacity; 8-32 all work
LORA_ALPHA = 32          # the adapter's strength is alpha / rank
LORA_DROPOUT = 0.05
MAX_LENGTH = 2048        # tokens per example; longer ones are cut off
BATCH_SIZE = 1           # per step and GPU; raise it if you have memory to spare
GRAD_ACCUMULATION = 16   # effective batch size = BATCH_SIZE x GRAD_ACCUMULATION
MIN_EXAMPLES = 20        # with fewer, the model would only memorise them


def load_examples(path: Path) -> list[dict]:
    """The chats in sft.jsonl that end with an assistant answer."""
    if not path.is_file():
        sys.exit(f"{path} not found - run this script in the folder that contains sft.jsonl.")
    examples = []
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                messages = json.loads(line)["messages"]
                if messages[-1]["role"] == "assistant":
                    examples.append({"messages": messages})
            except (ValueError, KeyError, IndexError, TypeError):
                sys.exit(f"{path.name}, line {number}: expected a JSON object with a non-empty 'messages' list.")
    if len(examples) < MIN_EXAMPLES:
        sys.exit(f"Only {len(examples)} usable examples (need at least {MIN_EXAMPLES}). Approve more in Duckstoken.")
    too_long = sum(1 for e in examples if sum(len(m.get("content") or "") for m in e["messages"]) > MAX_LENGTH * 3)
    if too_long:
        print(f"Note: about {too_long} example(s) are longer than {MAX_LENGTH} tokens and will be cut off.")
    return examples


def require_gpu(torch) -> None:
    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU found. This script needs an NVIDIA GPU with 48-80 GB of memory.")
    total = sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count()))
    gib = total / 2**30
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.device_count()} device(s), {gib:.0f} GiB in total)")
    if gib < 44:
        print("WARNING: training gpt-oss-20b usually needs 48 GB or more; expect an out-of-memory error.")


def main() -> None:
    examples = load_examples(DATA_FILE)
    print(f"{len(examples)} training examples")

    # Heavy imports come after the cheap checks above so that mistakes show up quickly.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
    from trl import SFTConfig, SFTTrainer

    require_gpu(torch)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # The model is published as 4-bit MXFP4. dequantize=True expands it to bf16, which training needs.
    # "eager" attention is what OpenAI's fine-tuning guide uses for gpt-oss (attention sinks).
    # On transformers older than 4.56 write torch_dtype= instead of dtype=.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        attn_implementation="eager",
        dtype=torch.bfloat16,
        quantization_config=Mxfp4Config(dequantize=True),
        use_cache=False,          # incompatible with gradient checkpointing
        device_map="auto",
    )

    # LoRA: small trainable matrices next to the frozen weights. "all-linear" covers the attention
    # projections; gpt-oss keeps its mixture-of-experts weights as plain parameters, which this does
    # not touch (OpenAI's guide shows how to add them with target_parameters if you want that).
    lora = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora)
    peft_model.print_trainable_parameters()

    args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        gradient_checkpointing=True,     # trades speed for a lot of memory
        bf16=True,
        max_length=MAX_LENGTH,
        packing=False,                   # one chat per training sequence
        logging_steps=5,
        save_strategy="no",              # only the final adapter is kept
        report_to="none",
    )

    # TRL applies the model's own chat template to every {"messages": [...]} row. An optional
    # "thinking" field on an assistant message (present when the export included reasoning) is
    # understood by gpt-oss' template. Note that the loss covers the whole chat, not just the answer.
    trainer = SFTTrainer(
        model=peft_model,
        args=args,
        train_dataset=Dataset.from_list(examples),
        processing_class=tokenizer,
    )
    trainer.train()

    # Saves only the adapter (a few hundred MB at most), not the 40 GB base model.
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"\nDone. The adapter is in {OUTPUT_DIR}")
    print("Next: convert it to GGUF with llama.cpp's convert_lora_to_gguf.py (see the top of this file),")
    print("then pick the .gguf file in Duckstoken under Settings -> 'LoRA adapter (.gguf)'.")


if __name__ == "__main__":
    main()
'''
_README = r'''# Training a LoRA from your Duckstoken data

This bundle was exported from Duckstoken (Learning > Training data). Everything in it comes from your own
chats: answers you liked or corrected, and answers the big model wrote for you. Read this page before you
upload anything anywhere.

## What is in this bundle

| File | What it is |
|---|---|
| `sft.jsonl` | One chat per line: the conversation up to a question, then the answer you approved. This is what the training script learns from ("supervised fine-tuning"). |
| `pref.jsonl` | Pairs of a better and a worse answer to the same prompt (`chosen` / `rejected`). Meant for preference tuning such as DPO. The included script does not use it. |
| `manifest.json` | How many examples are in each file, where they came from, when this was exported and whether it was redacted. |
| `train_lora.py` | An example training script for a rented NVIDIA GPU. It is a template that has not been run or tested by the Duckstoken project. |

Every line is a standalone JSON object in UTF-8, so you can open the files in any editor and delete lines you do
not want.

## Before you upload anything: privacy

* The files contain your questions and the answers exactly as you approved them. If you did not tick the
  redaction option, they also contain whatever you typed: names, e-mail addresses, keys, file paths.
* Redaction replaces e-mail addresses, phone numbers, API keys and tokens, IPv4 addresses, card-like numbers
  and Windows/Linux home-folder names with placeholders such as `<email>`. It is automatic and **best effort**: it
  cannot recognise a name, an address or a password written in ordinary prose.
* Search the files for anything private (use your editor's search) and delete those lines before uploading.
* A rented GPU is someone else's computer. Do not put data on it that you would not want its owner to see, and
  delete the machine and its disk when you are done.

## What to expect

* A LoRA is a small add-on that nudges a model towards your style, tone, formats and recurring preferences, and
  towards facts about your own projects. It does **not** make a 20B model as capable as a 120B one. Expect
  narrow, modest gains.
* A few hundred good examples are a sensible start; a few dozen are not enough and the script refuses to run
  below 20. Quality matters more than quantity: reject examples whose answer is not what you want.
* Training can also make things worse (the model may repeat your examples or forget skills). Keep the original
  model, try the adapter on a handful of questions you know the answer to, and lower **LoRA strength** in
  Settings if it feels over-trained.
* Examples that were exported without reasoning can teach a reasoning model to think less. If that bothers you,
  export again with "include reasoning" ticked (only the model's own, unedited answers carry reasoning).
* The adapter only works with the model it was trained on. The script trains `openai/gpt-oss-20b`, so the result
  loads on gpt-oss-20b in Duckstoken and **not** on gpt-oss-120b or any other model.

## What it costs

You need an NVIDIA GPU with roughly 48-80 GB of memory: the 4-bit weights are expanded for training. Renting one
GPU of that size usually costs a few dollars per hour. For a few hundred short examples plan on an hour or two,
including the roughly 40 GB model download. These are rough, unmeasured estimates; check the provider's price
before you start and stop the machine when the run has finished.

## Step by step

1. Review `sft.jsonl` (and `pref.jsonl` if you want to use it elsewhere) and remove anything private.
2. Rent a GPU machine with 48-80 GB of memory and a PyTorch/CUDA image, and open a terminal on it.
3. Copy `sft.jsonl` and `train_lora.py` into one folder on that machine.
4. Install the libraries listed at the top of `train_lora.py`, then run `python train_lora.py`. It saves the
   adapter to a folder called `duckstoken-lora`.
5. Convert the adapter to GGUF with llama.cpp (the exact command is in the comment at the top of the script) and
   download the resulting `duckstoken-lora.gguf`. Then stop the rented machine.
6. In Duckstoken open Settings, find **LoRA adapter (.gguf)**, pick the file, and restart the model. Make sure the
   base model you run is gpt-oss-20b.

If a step fails, the script is a starting point, not a guarantee: the libraries it uses change their interfaces
often, so compare it with the current documentation of Hugging Face TRL and PEFT and OpenAI's article "How to
fine-tune gpt-oss".
'''
