"""Builds what Duckstoken adds to a chat request: smart prompt, memories, liked examples, web results.

Everything lands in one system message so small models see a single, compact block of context.
"""

from __future__ import annotations

import re

SMART_PROMPTS = {
    "concise": "Answer directly and briefly, without filler. If you are not sure, say so.",
    "careful": ("Think step by step before answering. Check facts, numbers and code before presenting them. If you "
                "are unsure or lack information, say so plainly instead of guessing, and use the available tools "
                "(search, memory, files) when they would help. Give the answer first, then brief reasoning. Keep it "
                "as short as the question allows."),
    "tutor": ("Explain like a patient tutor: start with the key idea in plain words, then walk through the steps with "
              "a small example, and end with one question to check understanding. Say so if you are unsure."),
}

_REMEMBER = re.compile(r"\bremember\b", re.I)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return ""


def user_texts(messages: list[dict]) -> list[str]:
    return [_text(m.get("content")) for m in messages if m.get("role") == "user"]


def retrieval_query(messages: list[dict]) -> str:
    """The last two user messages: follow-ups like 'and the second one?' still find the right memories."""
    texts = user_texts(messages)
    return " ".join(t[:500] for t in texts[-2:])


def augment(messages: list[dict], cfg: dict, memory, exchanges, web) -> tuple[list[dict], dict]:
    """Return (messages with one composed system message in front, info about what was added)."""
    info: dict = {"memory_ids": [], "example_ids": [], "lesson_ids": [], "web_sources": [], "remembered": []}
    parts: list[str] = []
    mode = cfg.get("smart_prompt", "off")
    if mode in SMART_PROMPTS:
        parts.append(SMART_PROMPTS[mode])
    if cfg.get("system_prompt", "").strip():
        parts.append(cfg["system_prompt"].strip())

    rest = list(messages)
    if rest and rest[0].get("role") == "system":  # a caller-supplied system message goes after ours
        parts.append(_text(rest[0].get("content")))
        rest = rest[1:]

    texts = user_texts(rest)
    last = texts[-1] if texts else ""
    query = retrieval_query(rest)

    if last and memory is not None and cfg.get("memory_enabled"):
        try:
            if _REMEMBER.search(last):
                for c in memory.candidates_from_text(last):
                    memory.add(c, kind="fact", source="user")
                    info["remembered"].append(c)
            ctx = memory.build_context(query, cfg["memory_inject_items"], cfg["memory_inject_chars"])
            if ctx["text"]:
                parts.append(ctx["text"])
                info["memory_ids"] = ctx["ids"]
        except Exception as e:  # noqa: BLE001 - never break chat because memory failed
            info["memory_error"] = str(e)

    if last and exchanges is not None and cfg.get("examples_enabled"):
        try:
            block = exchanges.build_block(query, cfg["examples_max"], cfg["examples_chars"],
                                          lessons=bool(cfg.get("lessons_enabled")))
            if block["text"]:
                parts.append(block["text"])
                info["example_ids"], info["lesson_ids"] = block["example_ids"], block["lesson_ids"]
        except Exception as e:  # noqa: BLE001
            info["examples_error"] = str(e)

    if last and web is not None and cfg.get("web_grounding") == "auto" and web.needs_web(last):
        g = web.ground(last, cfg)
        if g.get("text"):
            parts.append(g["text"])
            info["web_sources"] = [{"title": s.get("title", ""), "url": s.get("url", "")} for s in g["sources"]]
        if g.get("error"):
            info["web_error"] = g["error"]

    out = ([{"role": "system", "content": "\n\n".join(p for p in parts if p)}] if parts else []) + rest
    return out, info


# ---------------------------------------------------------------- refine (critique + rewrite)
def _transcript(messages: list[dict], limit: int = 6000) -> str:
    lines = [f"{m['role'].upper()}: {_text(m.get('content'))}" for m in messages
             if m.get("role") in ("user", "assistant") and _text(m.get("content"))]
    return "\n\n".join(lines)[-limit:]


def critique_messages(messages: list[dict], answer: str) -> list[dict]:
    return [
        {"role": "system", "content": "You are a strict reviewer. Find concrete problems in the draft answer: wrong or "
                                      "unsupported facts, arithmetic or logic errors, bugs, missing steps, unclear or "
                                      "padded writing. Reply with a short numbered list of problems and fixes. If the "
                                      "draft is already correct and clear, reply exactly: NO ISSUES."},
        {"role": "user", "content": f"Conversation:\n{_transcript(messages)}\n\nDraft answer:\n{answer}"}]


def rewrite_messages(messages: list[dict], answer: str, critique: str) -> list[dict]:
    return [
        {"role": "system", "content": "Rewrite the draft answer so it fixes every issue in the review. Keep what was "
                                      "already good. Output ONLY the improved answer, with no commentary about the review."},
        {"role": "user", "content": f"Conversation:\n{_transcript(messages)}\n\nDraft answer:\n{answer}\n\n"
                                    f"Review:\n{critique}"}]
