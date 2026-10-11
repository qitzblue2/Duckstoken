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
        marks = ",".join("?" * len(exchange_ids))  # only placeholders are formatted into the SQL
        rows = self.db.query(f"""SELECT id, exchange_id, kind, messages_json, chosen, rejected, source, status, edited
                                 FROM dataset_items WHERE exchange_id IN ({marks})""", exchange_ids)
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
        for start in range(0, len(stale), SQL_CHUNK):
            part = stale[start:start + SQL_CHUNK]
            self.db.execute(f"DELETE FROM dataset_items WHERE id IN ({','.join('?' * len(part))})", part)

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
            for start in range(0, len(wanted), SQL_CHUNK):
                part = wanted[start:start + SQL_CHUNK]
                updated += self.db.execute(f"UPDATE dataset_items SET status = ? WHERE id IN ({','.join('?' * len(part))})",
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
        `include_reasoning` adds the model's chain of thought to answers that are the model's own unedited answer.
        """
        statuses = ("approved",) if only_approved else ("approved", "pending")
        rows = self.db.query(f"SELECT {_ITEM_COLUMNS} FROM dataset_items "
                             f"WHERE status IN ({','.join('?' * len(statuses))}) ORDER BY id", statuses)
        scrub = _redact if redact else str
        sft, pref = [], []
        sources = {kind: dict.fromkeys(SOURCES, 0) for kind in KINDS}
        for row in rows:
            prompt = [{"role": m["role"], "content": scrub(m["content"])} for m in _stored_turns(row["messages_json"])]
            if not prompt:
                continue
            thoughts = self._thoughts(row) if include_reasoning else {}

            def reply(text: str) -> dict:
                message = {"role": "assistant", "content": scrub(text)}
                if _norm(text) in thoughts:
                    message["thinking"] = scrub(thoughts[_norm(text)])
                return message

            if row["kind"] == "sft":
                sft.append({"messages": [*prompt, reply(row["chosen"])]})
            elif row["rejected"]:
                pref.append({"prompt": prompt, "chosen": [reply(row["chosen"])], "rejected": [reply(row["rejected"])]})
            else:
                continue
            sources[row["kind"]][row["source"]] = sources[row["kind"]].get(row["source"], 0) + 1
        if not sft and not pref:
            raise ValueError("No approved training examples yet - approve some in Learning > Training data first."
                             if only_approved else "There are no training examples to export yet.")
        now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
        manifest = {"created": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "duckstoken_version": __version__,
                    "redacted": bool(redact), "only_approved": bool(only_approved),
                    "include_reasoning": bool(include_reasoning), "counts": {"sft": len(sft), "pref": len(pref)},
                    "sources": sources}
        files = [("sft.jsonl", _jsonl(sft)), ("pref.jsonl", _jsonl(pref)),
                 ("manifest.json", json.dumps(manifest, indent=2) + "\n"), ("README_TRAINING.md", training_readme())]
        if include_script:
            files.append(("train_lora.py", train_script()))
        return _zip(files, now)

    def _thoughts(self, row: dict) -> dict[str, str]:
        """{normalized answer: reasoning} for the exchange's original answer, when it recorded any reasoning."""
        exchange = self.exchanges.get(row["exchange_id"])
        if not exchange or not exchange.get("reasoning", "").strip() or not exchange.get("answer", "").strip():
            return {}
        return {_norm(exchange["answer"]): exchange["reasoning"].strip()}


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
    longest_word = max((len(w) for w in re.findall(r"[a-z]+", token)), default=0)
    symbols = sum(not c.isalnum() for c in token)
    mixed = any(c.islower() for c in token) and any(c.isupper() for c in token) and any(c.isdigit() for c in token)
    capitals = [i for i, c in enumerate(token[:-1]) if c.isupper()]
    camel_case = len(capitals) >= 4 and sum(token[i + 1].islower() for i in capitals) / len(capitals) > 0.7
    random_like = (mixed and not camel_case and longest_word < 10 and symbols <= len(token) // 8
                   and _entropy(token) >= 4.0)
    return "<secret>" if random_like else token


def _hex_blob(m: re.Match) -> str:
    token = m.group()
    return "<secret>" if re.search(r"\d", token) and re.search(r"[a-fA-F]", token) else token


_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_TOKEN_CHARS = r"A-Za-z0-9+/_\-"

# Applied in order; each entry is (pattern, replacement string or function).
_REDACTIONS: list[tuple[re.Pattern, object]] = [
    (re.compile(r"-----BEGIN [A-Z ]{0,30}PRIVATE KEY-----[\s\S]{0,8000}?-----END [A-Z ]{0,30}PRIVATE KEY-----"),
     "<secret>"),
    (re.compile(r"(?<![\w.+%\-])[\w.+%\-]{1,64}@(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.){1,8}[A-Za-z]{2,24}(?!\w)"),
     "<email>"),
    (re.compile("|".join((
        r"\bsk-(?:ant-)?[A-Za-z0-9_\-]{20,}",                        # OpenAI / Anthropic
        r"\bgh[pousr]_[A-Za-z0-9]{30,}", r"\bgithub_pat_[A-Za-z0-9_]{20,}",
        r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",                            # AWS access key ids
        r"\bxox[abprs]-[A-Za-z0-9\-]{10,}",                          # Slack
        r"\bAIza[0-9A-Za-z_\-]{35}",                                 # Google API keys
        r"\bhf_[A-Za-z0-9]{30,}",                                    # Hugging Face
        r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}",                   # Stripe
        r"\beyJ[A-Za-z0-9_\-]{5,}\.[A-Za-z0-9_\-]{5,}\.[A-Za-z0-9_\-]*",  # JWT
    ))), "<secret>"),
    (re.compile(r"\b([Bb]earer\s+)(?=[A-Za-z0-9._~+/\-]*\d)[A-Za-z0-9._~+/\-]{16,}=*"), r"\1<secret>"),
    (re.compile(r"(?<![\w.\-])\d(?:[ \-]?\d){12,18}(?!\w)"), _card),
    (re.compile(r"""(?<![\w.\-/])(?:
          \+[1-9]\d{0,2}[\s.\-]?(?:\(\d{1,4}\)[\s.\-]?)?\d{1,4}(?:[\s.\-]?\d{2,4}){1,4}
        | (?:\(\d{3}\)\s?|\d{3}[\s.\-])\d{3}[\s.\-]\d{4}
        | 0\d{1,4}(?:[\s.\-]\d{2,4}){2,4}
        )(?!\w|[.\-/]\d)""", re.X), _phone),
    (re.compile(rf"(?<![\w.])(?:{_OCTET}\.){{3}}{_OCTET}(?![\w]|\.\d)"),
     lambda m: m.group() if m.group() in ("127.0.0.1", "0.0.0.0") else "<ip>"),
    (re.compile(r"(?<![0-9A-Za-z])[0-9a-fA-F]{32,}(?![0-9A-Za-z])"), _hex_blob),
    (re.compile(rf"(?<![{_TOKEN_CHARS}])[{_TOKEN_CHARS}]{{32,}}={{0,2}}(?![{_TOKEN_CHARS}])"), _random_blob),
    (re.compile(r"(?i)\b([A-Z]:[\\/]Users[\\/])(?!(?:Public|Default|All Users)[\\/])(?!\s)[^\\/:*?\"<>|\r\n]{1,64}(?=[\\/])"),
     r"\1<user>"),
    (re.compile(r"(?<![\w.\-:~])((?:/home|/Users)/)[^/\s<>:*?\"|]{1,64}(?=/)"), r"\1<user>"),
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


_TRAIN_SCRIPT = ""
_README = ""
