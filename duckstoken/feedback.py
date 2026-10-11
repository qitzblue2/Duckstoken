"""Every answer the user has seen, with their rating, note and correction.

This table is what lets the assistant improve: liked or corrected answers come back as few-shot
examples for similar questions, disliked ones with a note become reminders (and, when a memory store
is attached, long-term "lesson" memories), and the whole record feeds the training dataset.
Matching is plain BM25 over the stored prompts, so it works without a model or an index.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from typing import TYPE_CHECKING

from .db import Database

if TYPE_CHECKING:
    from .memory import MemoryStore

MAX_TEXT = 200_000                    # answer / reasoning / prompt / correction / teacher answer
MAX_MESSAGES_BYTES = 2 * 1024 * 1024  # stored conversation; oldest messages are dropped first
MAX_NOTE = 2000
MAX_NAME = 200                        # chat id, model name, effort
MAX_SOURCES = 20
MAX_INT = 2**31                       # ids, indexes and offsets are clamped well inside SQLite's range
MAX_LIMIT = 200                       # rows per list() page
MAX_SCAN = 2000                       # candidates examined when looking for similar prompts
MATCH_CHARS = 2000                    # only the start of a prompt is used for matching
LESSON_K = 3
MIN_COVERAGE = 0.4                    # share of the query's words a prompt must contain
K1, B = 1.5, 0.75                     # BM25 parameters

EXAMPLES_HEAD = "Examples of answers this user liked for similar questions (match their style and level of detail):"
REMINDERS_HEAD = "Reminders from past feedback:"
_EXAMPLE_OVERHEAD = len("\n\nQ: \nA: ")

_COLUMNS = ("id, chat_id, msg_index, created, prompt, answer, reasoning, model, effort, tokens, tps, rating, note, "
            "correction, teacher_answer, teacher_model, used_tools, web_sources")

_UPSERT = """
INSERT INTO exchanges(chat_id, msg_index, created, prompt, messages_json, answer, reasoning, model, effort, tokens,
                      tps, used_tools, web_sources)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(chat_id, msg_index) DO UPDATE SET
  created = CASE WHEN answer = excluded.answer THEN created ELSE excluded.created END,
  prompt = excluded.prompt, messages_json = excluded.messages_json, answer = excluded.answer,
  reasoning = excluded.reasoning, model = excluded.model, effort = excluded.effort, tokens = excluded.tokens,
  tps = excluded.tps, used_tools = excluded.used_tools, web_sources = excluded.web_sources,
  rating = CASE WHEN answer = excluded.answer THEN rating ELSE 0 END,
  note = CASE WHEN answer = excluded.answer THEN note ELSE '' END,
  correction = CASE WHEN answer = excluded.answer THEN correction ELSE '' END,
  teacher_answer = CASE WHEN answer = excluded.answer THEN teacher_answer ELSE '' END,
  teacher_model = CASE WHEN answer = excluded.answer THEN teacher_model ELSE '' END
"""

_STOPWORDS = frozenset("""
a about above after again all also am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further get give had has have having he her here hers him
his how i if in into is it its just let like make me more most my no nor not now of off on once only or other our
ours out over own please same she should so some such than that the their theirs them then there these they this
those through to too under until up us very was we were what when where which while who whom why will with would
you your yours hi hello hey thanks thank ok okay tell show want need help
""".split())

_WORD = re.compile(r"[^\W_]+")


# ------------------------------------------------------------------ text helpers
def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit]


def _fit(text: str, limit: int) -> str:
    """`text` shortened with an ellipsis so that it is at most `limit` characters."""
    if limit <= 0:
        return ""
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def _arg(value, limit: int) -> str:
    return _clip("" if value is None else str(value).strip(), limit)


def _squash(text: str) -> str:
    return " ".join(text.split())


def _content_text(content) -> str:
    """Plain text of a message's content, which may be a string or a list of parts."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    texts = []
    for part in content:
        if isinstance(part, str):
            texts.append(part)
        elif isinstance(part, dict) and part.get("type", "text") == "text" and isinstance(part.get("text"), str):
            texts.append(part["text"])
    return "\n".join(texts)


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return _clip(_content_text(m.get("content")).strip(), MAX_TEXT)
    return ""


def _clean_message(message: dict) -> dict:
    out = {}
    for key, value in message.items():
        if key == "content":
            value = _content_text(value)
        out[str(key)] = _clip(value, MAX_TEXT) if isinstance(value, str) else value
    return out


def _dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _bounded_messages(messages) -> str:
    """JSON for the stored conversation: text-only content, at most MAX_MESSAGES_BYTES, oldest dropped first."""
    cleaned = [_clean_message(m) for m in messages if isinstance(m, dict) and isinstance(m.get("role"), str)]
    sizes = [len(_dumps(m).encode("utf-8")) + 1 for m in cleaned]
    total, start = sum(sizes) + 2, 0
    while total > MAX_MESSAGES_BYTES and len(cleaned) - start > 1:
        total -= sizes[start]
        start += 1
    kept = cleaned[start:]
    if total > MAX_MESSAGES_BYTES:  # a single message with oversized extra fields (e.g. tool calls)
        kept = [{"role": kept[0]["role"], "content": kept[0].get("content", "")}]
    return _dumps(kept)


def _clean_sources(sources) -> str:
    out = []
    for s in sources or []:
        url = s.get("url") if isinstance(s, dict) else None
        if isinstance(url, str) and re.match(r"https?://", url.strip(), re.I):
            out.append({"title": _arg(s.get("title"), 300), "url": _clip(url.strip(), 2000)})
        if len(out) >= MAX_SOURCES:
            break
    return _dumps(out)


def _loads_list(raw: str) -> list:
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return value if isinstance(value, list) else []


def _item(row: dict, include_messages: bool) -> dict:
    row["used_tools"] = bool(row["used_tools"])
    row["web_sources"] = _loads_list(row["web_sources"])
    if include_messages:
        row["messages"] = _loads_list(row.pop("messages_json"))
    return row


def _clamp(value, low: int, high: int) -> int:
    return max(low, min(int(value), high))


def _as_id(value) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return _clamp(value, -1, MAX_INT)  # -1 never matches a row
    except (TypeError, ValueError):
        return None


def _check_rating(rating) -> int:
    if isinstance(rating, bool) or not isinstance(rating, int) or rating not in (-1, 0, 1):
        raise ValueError("rating must be -1, 0 or 1")
    return rating


def _like(text: str) -> str:
    return "%" + text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def _lesson_text(prompt: str, note: str) -> str:
    hint, note = _squash(prompt), _squash(note)
    if not hint:
        return note
    if len(hint) > 120:
        hint = hint[:120].rstrip() + "…"
    return f"When asked something like “{hint}”: {note}"


# ------------------------------------------------------------------ matching
def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _tokens(text: str) -> list[str]:
    words = _WORD.findall(text[:MATCH_CHARS].casefold())
    return [_stem(w) for w in words if len(w) > 1 and w not in _STOPWORDS]


def _rank(query: str, texts: list[str]) -> list[tuple[float, int]]:
    """BM25 scores of `texts` against `query`, best first (ties keep their order), unrelated texts left out."""
    wanted = sorted(set(_tokens(query)))
    if not wanted or not texts:
        return []
    docs = [Counter(_tokens(t)) for t in texts]
    lengths = [sum(d.values()) for d in docs]
    average = sum(lengths) / len(docs) or 1.0
    df = {w: sum(1 for d in docs if w in d) for w in wanted}
    need = 1 if len(wanted) == 1 else max(2, math.ceil(MIN_COVERAGE * len(wanted)))
    ranked = []
    for i, doc in enumerate(docs):
        hits = [w for w in wanted if w in doc]
        if len(hits) < need:
            continue
        norm = K1 * (1 - B + B * lengths[i] / average)
        score = sum(math.log(1 + (len(docs) - df[w] + 0.5) / (df[w] + 0.5)) * doc[w] * (K1 + 1) / (doc[w] + norm)
                    for w in hits)
        ranked.append((score, i))
    return sorted(ranked, key=lambda r: -r[0])


def _sep_len(section: str) -> int:
    """Characters a non-empty section takes away from the room left for the other one (its length + blank line)."""
    return len(section) + 2 if section else 0


def _reminders(lessons: list[dict], limit: int) -> tuple[str, list[int]]:
    text, ids = REMINDERS_HEAD, []
    for lesson in lessons:
        room = limit - len(text) - 1
        if room < 20:
            break
        text += "\n" + _fit("- " + _lesson_text(lesson["prompt"], lesson["note"]), min(300, room))
        ids.append(lesson["id"])
    return (text, ids) if ids else ("", [])


class ExchangeStore:
    def __init__(self, db: Database, memory: MemoryStore | None = None):
        self.db = db
        self.memory = memory

    # ------------------------------------------------------------------ writing
    def record(self, chat_id, msg_index, messages, answer, reasoning="", model="", effort="", tokens=0, tps=0.0,
               used_tools=False, web_sources=None) -> dict:
        """Store (or replace) the answer at `msg_index` of a chat; feedback survives only if the answer is unchanged."""
        chat_id = _arg(chat_id, MAX_NAME + 1)
        if not chat_id or len(chat_id) > MAX_NAME:
            raise ValueError("chat_id must be 1-200 characters")
        if isinstance(msg_index, bool) or not isinstance(msg_index, int) or not 0 <= msg_index < MAX_INT:
            raise ValueError("msg_index must be a non-negative integer")
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        tps = float(tps or 0)
        params = (chat_id, msg_index, time.time(), _last_user_text([m for m in messages if isinstance(m, dict)]),
                  _bounded_messages(messages), _arg(answer, MAX_TEXT), _arg(reasoning, MAX_TEXT),
                  _arg(model, MAX_NAME), _arg(effort, MAX_NAME), max(0, int(tokens or 0)),
                  tps if math.isfinite(tps) else 0.0, 1 if used_tools else 0, _clean_sources(web_sources))
        with self.db.transaction():
            self.db.execute(_UPSERT, params)
            return self.find(chat_id, msg_index)

    def rate(self, id, rating, note="") -> dict | None:
        rating, note, id = _check_rating(rating), _arg(note, MAX_NOTE), _as_id(id)
        if id is None or not self.db.execute("UPDATE exchanges SET rating = ?, note = ? WHERE id = ?",
                                             (rating, note, id)).rowcount:
            return None
        item = self.get(id)
        if self.memory is not None and rating == -1 and note and item:
            self.memory.add(_lesson_text(item["prompt"], note), kind="lesson", source="feedback")
        return item

    def correct(self, id, correction) -> dict | None:
        return self._set(id, "correction = ?", (_arg(correction, MAX_TEXT),))

    def set_teacher(self, id, answer, model="") -> dict | None:
        return self._set(id, "teacher_answer = ?, teacher_model = ?", (_arg(answer, MAX_TEXT), _arg(model, MAX_NAME)))

    def _set(self, id, assignment: str, values: tuple) -> dict | None:
        id = _as_id(id)
        if id is None or not self.db.execute(f"UPDATE exchanges SET {assignment} WHERE id = ?",
                                             (*values, id)).rowcount:
            return None
        return self.get(id)

    def delete(self, id) -> bool:
        id = _as_id(id)
        return id is not None and self.db.execute("DELETE FROM exchanges WHERE id = ?", (id,)).rowcount > 0

    def delete_chat(self, chat_id) -> int:
        return self.db.execute("DELETE FROM exchanges WHERE chat_id = ?", (_arg(chat_id, MAX_NAME + 1),)).rowcount

    # ------------------------------------------------------------------ reading
    def get(self, id, include_messages=False) -> dict | None:
        id = _as_id(id)
        return None if id is None else self._one("id = ?", (id,), include_messages)

    def find(self, chat_id, msg_index) -> dict | None:
        return self._one("chat_id = ? AND msg_index = ?", (_arg(chat_id, MAX_NAME + 1), msg_index))

    def _one(self, where: str, params: tuple, include_messages=False) -> dict | None:
        cols = _COLUMNS + (", messages_json" if include_messages else "")
        row = self.db.one(f"SELECT {cols} FROM exchanges WHERE {where}", params)
        return _item(row, include_messages) if row else None

    def list(self, rating=None, q="", with_teacher=None, corrected=None, limit=50, offset=0,
             include_messages=False) -> dict:
        where, params = [], []
        if rating is not None:
            where.append("rating = ?")
            params.append(_check_rating(rating))
        q = (q or "").replace("\x00", "").strip()[:200]
        if q:
            where.append("(" + " OR ".join(f"{c} LIKE ? ESCAPE '\\'" for c in
                                          ("prompt", "answer", "correction", "note", "teacher_answer")) + ")")
            params += [_like(q)] * 5
        for flag, column in ((with_teacher, "teacher_answer"), (corrected, "correction")):
            if flag is not None:
                where.append(f"{column} {'!=' if flag else '='} ''")
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        total = self.db.scalar(f"SELECT count(*) FROM exchanges{clause}", params)
        cols = _COLUMNS + (", messages_json" if include_messages else "")
        rows = self.db.query(f"SELECT {cols} FROM exchanges{clause} ORDER BY created DESC, id DESC LIMIT ? OFFSET ?",
                             [*params, _clamp(limit, 1, MAX_LIMIT), _clamp(offset, 0, MAX_INT)])
        return {"items": [_item(r, include_messages) for r in rows], "total": total}

    def stats(self) -> dict:
        return self.db.one("""SELECT count(*) AS total, coalesce(sum(rating = 1), 0) AS up,
                                     coalesce(sum(rating = -1), 0) AS down,
                                     coalesce(sum(correction != ''), 0) AS corrected,
                                     coalesce(sum(teacher_answer != ''), 0) AS with_teacher FROM exchanges""")

    def recent_without_teacher(self, n, since=0.0) -> list[dict]:
        """Newest exchanges (any rating) that have a prompt but no big-model answer yet, with their messages."""
        n = _clamp(n, 0, MAX_SCAN)
        if n == 0:
            return []
        rows = self.db.query(f"""SELECT {_COLUMNS}, messages_json FROM exchanges
                                 WHERE prompt != '' AND teacher_answer = '' AND created >= ?
                                 ORDER BY created DESC, id DESC LIMIT ?""", (since, n))
        return [_item(r, True) for r in rows]

    # ------------------------------------------------------------------ learning from feedback
    def examples_for(self, query, k=2, max_chars=1500) -> list[dict]:
        """Liked or corrected answers to prompts similar to `query`: [{id, prompt, answer, score}]."""
        if k <= 0 or max_chars // k <= 0:
            return []
        rows = self.db.query("""
            SELECT id, prompt, best FROM (
              SELECT id, created, prompt, CASE WHEN correction != '' THEN correction ELSE answer END AS best
              FROM exchanges WHERE (rating = 1 OR correction != '') AND prompt != '')
            WHERE best != '' AND length(prompt) + length(best) <= ?
            ORDER BY created DESC, id DESC LIMIT ?""", (max_chars // k, MAX_SCAN))
        picked, seen = [], set()
        for score, i in _rank(query, [r["prompt"] for r in rows]):
            row = rows[i]
            prompt, answer = row["prompt"].strip(), row["best"].strip()
            key = _squash(prompt.casefold())
            if answer and key not in seen:
                seen.add(key)
                picked.append({"id": row["id"], "prompt": prompt, "answer": answer, "score": round(score, 3)})
                if len(picked) == k:
                    break
        return picked

    def lessons_for(self, query, k=LESSON_K) -> list[dict]:
        """Disliked answers with a note for prompts similar to `query`: [{id, prompt, note, score}]."""
        if k <= 0:
            return []
        rows = self.db.query("""SELECT id, substr(prompt, 1, ?) AS prompt, note FROM exchanges
                                WHERE rating = -1 AND note != '' AND prompt != ''
                                ORDER BY created DESC, id DESC LIMIT ?""", (MATCH_CHARS, MAX_SCAN))
        picked, seen = [], set()
        for score, i in _rank(query, [r["prompt"] for r in rows]):
            row = rows[i]
            key = (_squash(row["prompt"].casefold()), _squash(row["note"].casefold()))
            if key not in seen:
                seen.add(key)
                picked.append({"id": row["id"], "prompt": row["prompt"].strip(), "note": row["note"],
                               "score": round(score, 3)})
                if len(picked) == k:
                    break
        return picked

    def build_block(self, query, k, max_chars, lessons=True) -> dict:
        """Prompt text with similar liked answers and past reminders, never longer than `max_chars`."""
        found = self.lessons_for(query) if lessons and max_chars > 0 else []
        reminders, lesson_ids = _reminders(found, max_chars // 3 if k > 0 else max_chars)
        examples, example_ids = self._examples_section(query, k, max_chars - _sep_len(reminders))
        spare = max_chars - _sep_len(examples)
        if found and k > 0 and spare > max_chars // 3:  # the examples left room: let the reminders use it
            reminders, lesson_ids = _reminders(found, spare)
        return {"text": "\n\n".join(t for t in (examples, reminders) if t),
                "example_ids": example_ids, "lesson_ids": lesson_ids}

    def _examples_section(self, query, k: int, room: int) -> tuple[str, list[int]]:
        avail = room - len(EXAMPLES_HEAD) - _EXAMPLE_OVERHEAD * k
        if k <= 0 or avail < k:
            return "", []
        found = self.examples_for(query, k, avail)
        if not found:
            return "", []
        blocks = [f"Q: {e['prompt']}\nA: {e['answer']}" for e in found]
        return "\n\n".join([EXAMPLES_HEAD, *blocks]), [e["id"] for e in found]
