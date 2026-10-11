import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("DUCKSTOKEN_HOME", tempfile.mkdtemp(prefix="duck-ctx-"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from duckstoken import context, settings as st  # noqa: E402


class FakeMemory:
    def __init__(self):
        self.saved = []

    def candidates_from_text(self, text):
        return ["The user's name is Duck"] if "remember" in text.lower() else []

    def add(self, text, kind="fact", source="user", **kw):
        self.saved.append((text, kind, source))

    def build_context(self, query, max_items, max_chars):
        self.last_query = query
        return {"text": "MEMORY BLOCK", "ids": [7], "items": []}


class FakeExchanges:
    def build_block(self, query, k, max_chars, lessons=True):
        return {"text": "EXAMPLES BLOCK", "example_ids": [1, 2], "lesson_ids": [3]}


class FakeWeb:
    def needs_web(self, q):
        return "latest" in q

    def ground(self, q, cfg):
        return {"text": "WEB BLOCK", "sources": [{"title": "T", "url": "https://x.test", "snippet": "s"}]}


def cfg(**kw):
    return dict(st.DEFAULTS, **kw)


class AugmentTests(unittest.TestCase):
    def test_everything_composed_into_one_system_message(self):
        m, info = context.augment([{"role": "user", "content": "what is the latest python?"}],
                                  cfg(web_grounding="auto", system_prompt="Be nice"), FakeMemory(), FakeExchanges(), FakeWeb())
        self.assertEqual(m[0]["role"], "system")
        sys_text = m[0]["content"]
        for part in ("Think step by step", "Be nice", "MEMORY BLOCK", "EXAMPLES BLOCK", "WEB BLOCK"):
            self.assertIn(part, sys_text)
        self.assertEqual(sys_text.count("\n\n"), 4)
        self.assertEqual([x["role"] for x in m], ["system", "user"])
        self.assertEqual((info["memory_ids"], info["example_ids"], info["lesson_ids"]), ([7], [1, 2], [3]))
        self.assertEqual(info["web_sources"][0]["url"], "https://x.test")

    def test_everything_off_adds_nothing(self):
        msgs = [{"role": "user", "content": "hi"}]
        m, info = context.augment(msgs, cfg(smart_prompt="off", memory_enabled=False, examples_enabled=False),
                                  FakeMemory(), FakeExchanges(), FakeWeb())
        self.assertEqual(m, msgs)
        self.assertEqual(info["memory_ids"], [])

    def test_web_only_in_auto_mode_and_when_needed(self):
        for mode, text, expect in (("off", "latest news", False), ("tool", "latest news", False),
                                   ("auto", "write a poem", False), ("auto", "latest news", True)):
            _, info = context.augment([{"role": "user", "content": text}], cfg(web_grounding=mode),
                                      FakeMemory(), FakeExchanges(), FakeWeb())
            self.assertEqual(bool(info["web_sources"]), expect, (mode, text))

    def test_explicit_remember_saves_once_per_candidate(self):
        mem = FakeMemory()
        _, info = context.augment([{"role": "user", "content": "Please remember my name is Duck"}], cfg(), mem, None, None)
        self.assertEqual(mem.saved, [("The user's name is Duck", "fact", "user")])
        self.assertEqual(info["remembered"], ["The user's name is Duck"])
        mem2 = FakeMemory()
        context.augment([{"role": "user", "content": "hello there"}], cfg(), mem2, None, None)
        self.assertEqual(mem2.saved, [])

    def test_retrieval_query_uses_last_two_user_messages_and_list_content(self):
        msgs = [{"role": "user", "content": "tell me about tea"}, {"role": "assistant", "content": "ok"},
                {"role": "user", "content": [{"type": "text", "text": "and coffee?"}]}]
        self.assertEqual(context.retrieval_query(msgs), "tell me about tea and coffee?")

    def test_existing_system_message_is_merged_after_ours(self):
        m, _ = context.augment([{"role": "system", "content": "CALLER"}, {"role": "user", "content": "hi"}],
                               cfg(), None, None, None)
        self.assertEqual(len([x for x in m if x["role"] == "system"]), 1)
        self.assertTrue(m[0]["content"].endswith("CALLER"))

    def test_component_failure_does_not_break_chat(self):
        class Boom(FakeMemory):
            def build_context(self, *a):
                raise RuntimeError("db locked")
        m, info = context.augment([{"role": "user", "content": "hi"}], cfg(), Boom(), None, None)
        self.assertIn("db locked", info["memory_error"])
        self.assertEqual(m[-1]["content"], "hi")

    def test_refine_prompts_include_conversation_and_draft(self):
        msgs = [{"role": "user", "content": "2+2?"}]
        c = context.critique_messages(msgs, "5")
        self.assertIn("USER: 2+2?", c[1]["content"])
        self.assertIn("Draft answer:\n5", c[1]["content"])
        r = context.rewrite_messages(msgs, "5", "wrong")
        self.assertIn("Review:\nwrong", r[1]["content"])


if __name__ == "__main__":
    unittest.main()
