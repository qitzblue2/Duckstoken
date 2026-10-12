"""HTTP endpoints for memory, feedback, the big-model queue, training data, web tools and answer refinement.

`LearningApi` is mixed into the request handler in app.py. `bind(app)` hands it the application object.
"""

from __future__ import annotations

import json
import time

from . import context, llm, websearch
from .errors import ApiError
from .memory import extraction_messages, parse_extraction

APP = None


def bind(app) -> None:
    global APP
    APP = app


def post(fn):
    fn._method = "POST"
    return fn


def _int(q: dict, name: str, default: int, lo: int = 0, hi: int = 10**9) -> int:
    try:
        return max(lo, min(hi, int(q.get(name, default))))
    except (TypeError, ValueError):
        return default


def _tri(q: dict, name: str):
    """'1' -> True, '0' -> False, anything else/absent -> None."""
    v = q.get(name, "")
    return True if v == "1" else False if v == "0" else None


def _need(b: dict, name: str, typ=None):
    v = b.get(name)
    if v is None or (typ is not None and not isinstance(v, typ)):
        raise ApiError(f"'{name}' is required")
    return v


def _need_item(item, what: str = "item"):
    if item is None:
        raise ApiError(f"No such {what}", 404)
    return item


def _last_user_prompt(messages: list[dict]) -> str:
    texts = context.user_texts(messages)
    return texts[-1] if texts else ""


class LearningApi:
    # ------------------------------------------------------------------ memory
    def api_memory_list(self, q, b):
        return APP.memory.list(q.get("q", ""), q.get("kind", ""), _int(q, "limit", 50, 1, 500), _int(q, "offset", 0))

    def api_memory_stats(self, q, b):
        return APP.memory.stats()

    def api_memory_preview(self, q, b):
        cfg, query = APP.settings.all(), q.get("q", "")
        items = APP.memory.search(query, k=cfg["memory_inject_items"])
        ctx = APP.memory.build_context(query, cfg["memory_inject_items"], cfg["memory_inject_chars"])
        examples = APP.exchanges.build_block(query, cfg["examples_max"], cfg["examples_chars"],
                                             lessons=bool(cfg["lessons_enabled"]))
        return {"text": ctx["text"], "ids": ctx["ids"], "items": items, "examples": examples,
                "web_would_search": bool(cfg["web_grounding"] == "auto" and websearch.needs_web(query))}

    @post
    def api_memory_add(self, q, b):
        try:
            return {"item": APP.memory.add(_need(b, "text", str), kind=b.get("kind", "fact"), source="user",
                                           url=b.get("url", ""), title=b.get("title", ""),
                                           pinned=bool(b.get("pinned")))}
        except ValueError as e:
            raise ApiError(str(e)) from e

    @post
    def api_memory_update(self, q, b):
        fields = {k: b[k] for k in ("text", "kind", "pinned", "enabled", "title", "url") if k in b}
        try:
            return {"item": _need_item(APP.memory.update(_need(b, "id", int), **fields), "memory")}
        except ValueError as e:
            raise ApiError(str(e)) from e

    @post
    def api_memory_delete(self, q, b):
        return {"ok": APP.memory.delete(_need(b, "id", int))}

    @post
    def api_memory_clear(self, q, b):
        return {"deleted": APP.memory.clear(b.get("kind") or None)}

    @post
    def api_memory_add_doc(self, q, b):
        cfg = APP.settings.all()
        url, title, text = (b.get("url") or "").strip(), (b.get("title") or "").strip(), b.get("text") or ""
        try:
            if url and not text:
                page = websearch.fetch_page(url, cfg, max_chars=200_000)
                title, text = title or page["title"] or url, page["text"]
            if not text.strip():
                raise ApiError("Give me some text or a URL")
            return {"chunks": APP.memory.add_document(title or "Untitled", text, source_url=url), "title": title}
        except websearch.WebError as e:
            raise ApiError(str(e)) from e
        except ValueError as e:
            raise ApiError(str(e)) from e

    def api_memory_export(self, q, b):
        return {"items": APP.memory.export_all()}

    @post
    def api_memory_import(self, q, b):
        return {"added": APP.memory.import_items(b.get("items") or [])}

    @post
    def api_memory_extract(self, q, b):
        cfg = APP.settings.all()
        msgs = [m for m in (b.get("messages") or []) if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
        if not msgs:
            raise ApiError("No conversation to look at")
        try:
            res = llm.complete(APP.server, extraction_messages(msgs, cfg["memory_extract_max"]), max_tokens=900,
                               temperature=0.2, effort="low", timeout=cfg["server_timeout"])
        except llm.LLMError as e:
            raise ApiError(str(e), 503) from e
        return {"candidates": parse_extraction(res["content"], cfg["memory_extract_max"])}

    # ---------------------------------------------------------------- feedback
    @post
    def api_feedback_record(self, q, b):
        cfg = APP.settings.all()
        if not cfg["record_exchanges"]:
            return {"id": None}
        messages = [m for m in (b.get("messages") or []) if isinstance(m, dict)]
        answer = _need(b, "answer", str)
        ex = APP.exchanges.record(_need(b, "chat_id", str), _need(b, "msg_index", int), messages, answer,
                                  reasoning=b.get("reasoning", ""), model=b.get("model", ""),
                                  effort=b.get("effort", ""), tokens=int(b.get("tokens") or 0),
                                  tps=float(b.get("tps") or 0), used_tools=bool(b.get("used_tools")),
                                  web_sources=b.get("web_sources"))
        if cfg["auto_escalate_hard"] and cfg["big_model_path"] and ex["rating"] == 0 and not ex["teacher_answer"]:
            from .escalation import hard_prompt, unsure_answer
            if hard_prompt(messages) or unsure_answer(answer):
                self._enqueue(ex, "hard", cfg)
        return {"id": ex["id"]}

    def api_feedback_list(self, q, b):
        rating = q.get("rating", "")
        return APP.exchanges.list(rating=int(rating) if rating in ("1", "-1", "0") else None, q=q.get("q", ""),
                                  with_teacher=_tri(q, "with_teacher"), corrected=_tri(q, "corrected"),
                                  limit=_int(q, "limit", 50, 1, 200), offset=_int(q, "offset", 0))

    def api_feedback_get(self, q, b):
        return {"item": _need_item(APP.exchanges.get(_int(q, "id", 0), include_messages=True), "answer")}

    def api_feedback_stats(self, q, b):
        return APP.exchanges.stats()

    @post
    def api_feedback_rate(self, q, b):
        cfg = APP.settings.all()
        try:
            item = _need_item(APP.exchanges.rate(_need(b, "id", int), int(b.get("rating", 0)), b.get("note", "")),
                              "answer")
        except ValueError as e:
            raise ApiError(str(e)) from e
        if item["rating"] == -1 and cfg["auto_escalate_downvote"] and cfg["big_model_path"]:
            self._enqueue(item, "downvote", cfg)
        return {"item": item}

    @post
    def api_feedback_correct(self, q, b):
        return {"item": _need_item(APP.exchanges.correct(_need(b, "id", int), b.get("correction", "")), "answer")}

    @post
    def api_feedback_delete(self, q, b):
        return {"ok": APP.exchanges.delete(_need(b, "id", int))}

    @post
    def api_feedback_delete_chat(self, q, b):
        """The chat was deleted in the UI: remove what was logged from it (answers, queued jobs, training examples)."""
        return APP.exchanges.purge_chat(_need(b, "chat_id", str))

    # ------------------------------------------------------------------- queue
    def _enqueue(self, exchange: dict, reason: str, cfg: dict):
        full = APP.exchanges.get(exchange["id"], include_messages=True)
        try:
            job = APP.queue.enqueue(full["messages"], chat_id=full["chat_id"], exchange_id=full["id"],
                                    reason=reason, max_queued=cfg["queue_max"])
        except ValueError as e:
            raise ApiError(str(e)) from e
        APP.worker.wake()
        return job

    def _job_view(self, job: dict) -> dict:
        out = dict(job)
        msgs = json.loads(out.pop("messages_json", "[]") or "[]")
        out["prompt"] = _last_user_prompt(msgs)[:300]
        return out

    def api_queue_status(self, q, b):
        cfg = APP.settings.all()
        s = APP.worker.status()
        s.update(run_mode=cfg["big_run_mode"], big_model_path=cfg["big_model_path"],
                 overnight={"enabled": cfg["big_run_mode"] == "overnight" or cfg["overnight_review_count"] > 0,
                            "start": cfg["overnight_start"], "end": cfg["overnight_end"],
                            "review_count": cfg["overnight_review_count"]})
        return s

    def api_queue_list(self, q, b):
        items = APP.queue.list(status=q.get("status") or None, limit=_int(q, "limit", 100, 1, 500))
        return {"items": [self._job_view(j) for j in items]}

    def api_queue_get(self, q, b):
        job = _need_item(APP.queue.get(_int(q, "id", 0)), "job")
        view = self._job_view(job)
        view["messages"] = json.loads(job.get("messages_json") or "[]")
        ex = APP.exchanges.get(job["exchange_id"]) if job.get("exchange_id") else None
        view["small_answer"] = ex["answer"] if ex else ""
        return {"item": view}

    @post
    def api_queue_enqueue(self, q, b):
        cfg = APP.settings.all()
        if not cfg["big_model_path"]:
            raise ApiError("Pick a big model first (Settings → Big model, or Learning → Big model).")
        reason = b.get("reason") if b.get("reason") in ("manual", "downvote", "hard", "review") else "manual"
        if b.get("exchange_id") is not None:
            ex = _need_item(APP.exchanges.get(int(b["exchange_id"])), "answer")
            return {"item": self._view_enqueue(self._enqueue(ex, reason, cfg))}
        try:
            job = APP.queue.enqueue([m for m in (b.get("messages") or []) if isinstance(m, dict)],
                                    chat_id=str(b.get("chat_id") or ""), reason=reason, max_queued=cfg["queue_max"])
        except ValueError as e:
            raise ApiError(str(e)) from e
        APP.worker.wake()
        return {"item": self._view_enqueue(job)}

    def _view_enqueue(self, job: dict) -> dict:
        return self._job_view(job)

    @post
    def api_queue_cancel(self, q, b):
        return {"ok": APP.queue.cancel(_need(b, "id", int))}

    @post
    def api_queue_clear(self, q, b):
        return {"removed": APP.queue.clear_finished()}

    @post
    def api_queue_run_now(self, q, b):
        if not APP.settings.get("big_model_path"):
            raise ApiError("Pick a big model first.")
        APP.worker.run_now()
        return {"ok": True}

    @post
    def api_queue_review(self, q, b):
        if not APP.settings.get("big_model_path"):
            raise ApiError("Pick a big model first.")
        n = APP.worker.schedule_review(max(1, min(200, int(b.get("n") or 10))))
        APP.worker.wake()
        return {"queued": n}

    # ----------------------------------------------------------------- dataset
    @post
    def api_dataset_refresh(self, q, b):
        return APP.dataset.refresh()

    def api_dataset_list(self, q, b):
        return APP.dataset.list(status=q.get("status", ""), kind=q.get("kind", ""), q=q.get("q", ""),
                                limit=_int(q, "limit", 50, 1, 200), offset=_int(q, "offset", 0))

    def api_dataset_stats(self, q, b):
        return APP.dataset.stats()

    @post
    def api_dataset_set_status(self, q, b):
        try:
            return {"updated": APP.dataset.set_status([int(i) for i in b.get("ids") or []], b.get("status", ""))}
        except (ValueError, TypeError) as e:
            raise ApiError(str(e)) from e

    @post
    def api_dataset_edit(self, q, b):
        return {"item": _need_item(APP.dataset.edit(_need(b, "id", int), _need(b, "chosen", str), b.get("rejected")),
                                   "training example")}

    @post
    def api_dataset_delete(self, q, b):
        return {"ok": APP.dataset.delete(_need(b, "id", int))}

    def api_dataset_export(self, q, b):
        cfg = APP.settings.all()
        try:
            blob = APP.dataset.export_zip(only_approved=q.get("only_approved", "1") == "1",
                                          redact=q.get("redact", "1" if cfg["redact_pii"] else "0") == "1",
                                          include_reasoning=q.get("include_reasoning") == "1",
                                          include_script=q.get("include_script", "1") == "1")
        except ValueError as e:
            raise ApiError(str(e)) from e
        self._send(200, blob, "application/zip",
                   {"Content-Disposition": f'attachment; filename="duckstoken-training-{time.strftime("%Y%m%d")}.zip"'})

    # --------------------------------------------------------------------- web
    @post
    def api_web_test(self, q, b):
        cfg = APP.settings.all()
        t0 = time.time()
        try:
            results = websearch.search(str(b.get("query") or "duckstoken"), cfg)
        except websearch.WebError as e:
            raise ApiError(str(e)) from e
        return {"results": results, "provider": cfg["web_provider"], "seconds": round(time.time() - t0, 1)}

    # ------------------------------------------------------------------ refine
    @post
    def api_refine(self, q, b):
        cfg = APP.settings.all()
        msgs = [m for m in (b.get("messages") or []) if isinstance(m, dict)]
        answer = _need(b, "answer", str)
        cap = cfg["max_tokens"] if cfg["max_tokens"] > 0 else 2048
        critique_all = []
        try:
            for _ in range(cfg["reflect_passes"]):
                crit = llm.complete(APP.server, context.critique_messages(msgs, answer), max_tokens=700,
                                    temperature=0.2, effort=cfg["reasoning_effort"], timeout=cfg["server_timeout"])
                c = crit["content"].strip()
                critique_all.append(c)
                if not c or c.upper().startswith("NO ISSUES"):
                    break
                answer = llm.complete(APP.server, context.rewrite_messages(msgs, answer, c), max_tokens=cap,
                                      temperature=cfg["temperature"], effort=cfg["reasoning_effort"],
                                      timeout=cfg["server_timeout"])["content"].strip() or answer
        except llm.LLMError as e:
            raise ApiError(str(e), 503) from e
        return {"answer": answer, "critique": "\n\n".join(critique_all)}
