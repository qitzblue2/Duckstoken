import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["DUCKSTOKEN_HOME"] = tempfile.mkdtemp(prefix="duck-feedback-")

from duckstoken import feedback  # noqa: E402
from duckstoken.db import Database  # noqa: E402
from duckstoken.feedback import ExchangeStore  # noqa: E402


class FakeMemory:
    def __init__(self):
        self.added = []

    def add(self, text, kind="fact", source="user", **_):
        self.added.append({"text": text, "kind": kind, "source": source})
        return self.added[-1]


def convo(prompt, *earlier):
    return [*earlier, {"role": "user", "content": prompt}]


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = self.open_db("x.db")
        self.memory = FakeMemory()
        self.store = ExchangeStore(self.db, self.memory)
        self.n = 0

    def open_db(self, name):
        db = Database(Path(self.tmp.name) / name)
        self.addCleanup(db.close)  # Windows cannot delete a database that is still open
        return db

    def add(self, prompt, answer="An answer.", chat="c1", **kw):
        self.n += 1
        return self.store.record(chat, self.n, convo(prompt), answer, **kw)

    def liked(self, prompt, answer="An answer.", **kw):
        item = self.add(prompt, answer, **kw)
        return self.store.rate(item["id"], 1)

    def disliked(self, prompt, note, answer="Bad answer."):
        item = self.add(prompt, answer)
        return self.store.rate(item["id"], -1, note)


class RecordTests(StoreCase):
    def test_stores_all_fields(self):
        item = self.store.record("c1", 3, convo("What is Python?"), "A language.", reasoning="thinking", model="m",
                                 effort="high", tokens=42, tps=12.5, used_tools=True,
                                 web_sources=[{"title": "Py", "url": "https://python.org", "snippet": "x"}])
        self.assertEqual((item["chat_id"], item["msg_index"], item["prompt"]), ("c1", 3, "What is Python?"))
        self.assertEqual((item["answer"], item["reasoning"], item["model"], item["effort"]),
                         ("A language.", "thinking", "m", "high"))
        self.assertEqual((item["tokens"], item["tps"], item["used_tools"]), (42, 12.5, True))
        self.assertEqual(item["web_sources"], [{"title": "Py", "url": "https://python.org"}])
        self.assertEqual((item["rating"], item["note"], item["correction"], item["teacher_answer"]), (0, "", "", ""))
        self.assertNotIn("messages", item)
        self.assertNotIn("messages_json", item)
        self.assertGreater(item["created"], 0)

    def test_messages_only_when_requested(self):
        msgs = [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hi there"}]
        item = self.store.record("c1", 1, msgs, "hello")
        self.assertEqual(self.store.get(item["id"], include_messages=True)["messages"], msgs)
        self.assertNotIn("messages", self.store.get(item["id"]))
        self.assertNotIn("messages_json", self.store.get(item["id"], include_messages=True))
        self.assertEqual(self.store.find("c1", 1)["id"], item["id"])
        self.assertIsNone(self.store.find("c1", 2))
        self.assertIsNone(self.store.find("other", 1))

    def test_prompt_is_last_user_message(self):
        msgs = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "  second  "}, {"role": "tool", "content": "out", "tool_call_id": "t"}]
        self.assertEqual(self.store.record("c", 0, msgs, "a")["prompt"], "second")
        self.assertEqual(self.store.record("c", 1, [{"role": "system", "content": "s"}], "a")["prompt"], "")

    def test_content_as_list_of_parts(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "Describe"},
                                             {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                                             "this", {"type": "text", "text": "photo"}]}]
        item = self.store.record("c", 0, msgs, "a", )
        self.assertEqual(item["prompt"], "Describe\nthis\nphoto")
        stored = self.store.get(item["id"], include_messages=True)["messages"]
        self.assertEqual(stored, [{"role": "user", "content": "Describe\nthis\nphoto"}])
        self.assertEqual(self.store.record("c", 1, [{"role": "user", "content": 5}], "a")["prompt"], "")

    def test_extra_message_keys_survive(self):
        msgs = [{"role": "assistant", "content": None, "tool_calls": [{"id": "1", "function": {"name": "f"}}]},
                {"role": "tool", "tool_call_id": "1", "name": "f", "content": "result"},
                {"role": "user", "content": "q"}]
        item = self.store.record("c", 0, msgs, "a")
        stored = self.store.get(item["id"], include_messages=True)["messages"]
        self.assertEqual(stored[0]["tool_calls"][0]["id"], "1")
        self.assertEqual(stored[0]["content"], "")
        self.assertEqual(stored[1]["tool_call_id"], "1")

    def test_invalid_messages_are_ignored(self):
        item = self.store.record("c", 0, ["junk", None, {"content": "no role"}, {"role": "user", "content": "q"}], "a")
        self.assertEqual(self.store.get(item["id"], include_messages=True)["messages"],
                         [{"role": "user", "content": "q"}])

    def test_upsert_keeps_id_and_replaces_stats(self):
        first = self.store.record("c", 1, convo("q"), "same", reasoning="r1", model="m1", tokens=5, tps=1.0)
        second = self.store.record("c", 1, convo("q"), "same", reasoning="r2", model="m2", tokens=9, tps=2.0,
                                   used_tools=True, web_sources=[{"title": "t", "url": "http://x.org"}])
        self.assertEqual(second["id"], first["id"])
        self.assertEqual((second["reasoning"], second["model"], second["tokens"], second["tps"], second["used_tools"]),
                         ("r2", "m2", 9, 2.0, True))
        self.assertEqual(second["created"], first["created"])
        self.assertEqual(self.store.stats()["total"], 1)

    def test_upsert_keeps_feedback_when_answer_unchanged(self):
        item = self.store.record("c", 1, convo("q"), "same")
        self.store.rate(item["id"], -1, "wrong")
        self.store.correct(item["id"], "right")
        self.store.set_teacher(item["id"], "teacher says", "big")
        again = self.store.record("c", 1, convo("q"), "same", tokens=7)
        self.assertEqual((again["rating"], again["note"], again["correction"]), (-1, "wrong", "right"))
        self.assertEqual((again["teacher_answer"], again["teacher_model"]), ("teacher says", "big"))
        self.assertEqual(again["tokens"], 7)

    def test_upsert_resets_feedback_when_answer_changes(self):
        item = self.store.record("c", 1, convo("q"), "old")
        self.store.rate(item["id"], 1, "nice")
        self.store.correct(item["id"], "fixed")
        self.store.set_teacher(item["id"], "teacher says", "big")
        time.sleep(0.01)
        again = self.store.record("c", 1, convo("q"), "new")
        self.assertEqual(again["id"], item["id"])
        self.assertEqual((again["answer"], again["rating"], again["note"], again["correction"]), ("new", 0, "", ""))
        self.assertEqual((again["teacher_answer"], again["teacher_model"]), ("", ""))
        self.assertGreater(again["created"], item["created"])

    def test_same_index_in_other_chat_is_separate(self):
        a = self.store.record("a", 1, convo("q"), "x")
        b = self.store.record("b", 1, convo("q"), "x")
        self.assertNotEqual(a["id"], b["id"])

    def test_answers_are_stripped_and_bounded(self):
        item = self.store.record("c", 0, convo("q"), "  hi \n", reasoning="  ")
        self.assertEqual((item["answer"], item["reasoning"]), ("hi", ""))
        huge = self.store.record("c", 1, convo("q" * 300_000), "a" * 300_000, reasoning="r" * 300_000)
        self.assertEqual(len(huge["answer"]), feedback.MAX_TEXT)
        self.assertEqual(len(huge["reasoning"]), feedback.MAX_TEXT)
        self.assertEqual(len(huge["prompt"]), feedback.MAX_TEXT)

    def test_oversized_conversation_drops_oldest_messages(self):
        chunk = "x" * 150_000
        msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"{i}" + chunk} for i in range(40)]
        msgs.append({"role": "user", "content": "the question"})
        item = self.store.record("c", 0, msgs, "a")
        raw = self.db.scalar("SELECT messages_json FROM exchanges WHERE id = ?", (item["id"],))
        self.assertLessEqual(len(raw.encode("utf-8")), feedback.MAX_MESSAGES_BYTES)
        kept = json.loads(raw)
        self.assertEqual(kept[-1]["content"], "the question")
        self.assertTrue(kept[0]["content"].startswith("2") or kept[0]["content"].startswith("3"))  # newest survive
        self.assertGreater(len(kept), 5)
        self.assertEqual(item["prompt"], "the question")

    def test_single_oversized_message_is_reduced(self):
        msgs = [{"role": "assistant", "content": "", "tool_calls": [{"function": {"arguments": "z" * 3_000_000}}]}]
        item = self.store.record("c", 0, msgs, "a")
        raw = self.db.scalar("SELECT messages_json FROM exchanges WHERE id = ?", (item["id"],))
        self.assertLessEqual(len(raw.encode("utf-8")), feedback.MAX_MESSAGES_BYTES)
        self.assertEqual(json.loads(raw), [{"role": "assistant", "content": ""}])

    def test_multibyte_conversation_is_bounded_in_bytes(self):
        msgs = [{"role": "user", "content": "日本語" * 60_000} for _ in range(20)]
        item = self.store.record("c", 0, msgs, "a")
        raw = self.db.scalar("SELECT messages_json FROM exchanges WHERE id = ?", (item["id"],))
        self.assertLessEqual(len(raw.encode("utf-8")), feedback.MAX_MESSAGES_BYTES)

    def test_web_sources_are_sanitised(self):
        sources = [{"title": "ok", "url": "https://a.org/x"}, {"title": "bad", "url": "javascript:alert(1)"},
                   {"title": "nourl"}, "junk", {"title": "t" * 1000, "url": " HTTP://b.org "}]
        sources += [{"title": str(i), "url": f"https://s{i}.org"} for i in range(50)]
        item = self.store.record("c", 0, convo("q"), "a", web_sources=sources)
        urls = [s["url"] for s in item["web_sources"]]
        self.assertEqual(urls[:2], ["https://a.org/x", "HTTP://b.org"])
        self.assertEqual(len(urls), feedback.MAX_SOURCES)
        self.assertEqual(len(item["web_sources"][1]["title"]), 300)
        self.assertEqual(self.store.record("c", 1, convo("q"), "a", web_sources="nope")["web_sources"], [])

    def test_numeric_fields_are_coerced(self):
        item = self.store.record("c", 0, convo("q"), "a", tokens=-5, tps=float("nan"), used_tools=0)
        self.assertEqual((item["tokens"], item["tps"], item["used_tools"]), (0, 0.0, False))
        self.assertEqual(self.store.record("c", 1, convo("q"), "a", tokens=None, tps=None)["tokens"], 0)

    def test_validation(self):
        for chat_id in ("", "   ", "x" * 201):
            with self.assertRaises(ValueError):
                self.store.record(chat_id, 0, [], "a")
        for index in (-1, "1", 1.5, True, None):
            with self.assertRaises(ValueError):
                self.store.record("c", index, [], "a")
        with self.assertRaises(ValueError):
            self.store.record("c", 0, "not a list", "a")
        with self.assertRaises(ValueError):
            self.store.record("c", 10**30, [], "a")

    def test_sql_metacharacters_are_data(self):
        item = self.store.record("c'; DROP TABLE exchanges; --", 0, convo("q'); DELETE FROM exchanges; --"),
                                 "a\" OR 1=1 --")
        self.assertEqual(self.store.get(item["id"])["answer"], "a\" OR 1=1 --")
        self.assertEqual(self.store.stats()["total"], 1)

    def test_concurrent_records_do_not_duplicate(self):
        errors = []

        def work():
            try:
                for i in range(20):
                    self.store.record("c", i % 5, convo("q"), "a")
            except Exception as e:  # pragma: no cover - only on failure
                errors.append(e)

        threads = [threading.Thread(target=work) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(self.store.stats()["total"], 5)


class FeedbackTests(StoreCase):
    def test_rating_validation(self):
        item = self.add("q")
        for bad in (2, -2, "1", 1.0, True, None, False):
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.store.rate(item["id"], bad)
        self.assertEqual(self.store.rate(item["id"], 1)["rating"], 1)
        self.assertEqual(self.store.rate(item["id"], 0)["rating"], 0)

    def test_rate_unknown_or_bad_id(self):
        for bad in (9999, "abc", None, True, [1]):
            self.assertIsNone(self.store.rate(bad, 1))
        self.assertEqual(self.memory.added, [])

    def test_rate_stores_note_and_accepts_string_id(self):
        item = self.add("q")
        rated = self.store.rate(str(item["id"]), 1, "  nice and short  ")
        self.assertEqual((rated["rating"], rated["note"]), (1, "nice and short"))
        self.assertEqual(self.store.rate(item["id"], 1)["note"], "")
        self.assertEqual(len(self.store.rate(item["id"], 1, "n" * 5000)["note"]), feedback.MAX_NOTE)

    def test_downvote_with_note_creates_lesson(self):
        item = self.add("How do I reverse a list in Python?")
        self.store.rate(item["id"], -1, "Too long, just show the one-liner")
        self.assertEqual(self.memory.added, [{
            "text": "When asked something like “How do I reverse a list in Python?”: "
                    "Too long, just show the one-liner",
            "kind": "lesson", "source": "feedback"}])

    def test_lesson_prompt_is_cut_at_120_chars(self):
        item = self.add("word " * 100)
        self.store.rate(item["id"], -1, "be brief")
        text = self.memory.added[0]["text"]
        quoted = text.split("“")[1].split("”")[0]
        self.assertTrue(quoted.endswith("…"))
        self.assertLessEqual(len(quoted), 121)
        self.assertGreater(len(quoted), 100)
        self.assertTrue(text.endswith(": be brief"))

    def test_lesson_without_prompt_is_just_the_note(self):
        item = self.store.record("c", 0, [{"role": "system", "content": "s"}], "a")
        self.store.rate(item["id"], -1, "never do that")
        self.assertEqual(self.memory.added[0]["text"], "never do that")

    def test_no_lesson_without_note_or_for_other_ratings_or_memory(self):
        item = self.add("q")
        self.store.rate(item["id"], -1)
        self.store.rate(item["id"], -1, "   ")
        self.store.rate(item["id"], 1, "great")
        self.store.rate(item["id"], 0, "meh")
        self.assertEqual(self.memory.added, [])
        plain = ExchangeStore(self.db)
        self.assertEqual(plain.rate(item["id"], -1, "bad")["rating"], -1)

    def test_correct_set_and_clear_leaves_rating_alone(self):
        item = self.add("q")
        self.store.rate(item["id"], -1)
        fixed = self.store.correct(item["id"], "  the right answer \n")
        self.assertEqual((fixed["correction"], fixed["rating"]), ("the right answer", -1))
        self.assertEqual(self.store.correct(item["id"], "")["correction"], "")
        self.assertEqual(self.store.correct(item["id"], None)["correction"], "")
        self.assertIsNone(self.store.correct(9999, "x"))
        self.assertEqual(len(self.store.correct(item["id"], "c" * 300_000)["correction"]), feedback.MAX_TEXT)

    def test_set_teacher(self):
        item = self.add("q")
        taught = self.store.set_teacher(item["id"], "  big answer ", "gpt-oss-120b")
        self.assertEqual((taught["teacher_answer"], taught["teacher_model"]), ("big answer", "gpt-oss-120b"))
        self.assertEqual(self.store.set_teacher(item["id"], "again")["teacher_model"], "")
        self.assertIsNone(self.store.set_teacher(9999, "x"))

    def test_delete(self):
        item = self.add("q")
        self.assertTrue(self.store.delete(item["id"]))
        self.assertFalse(self.store.delete(item["id"]))
        self.assertFalse(self.store.delete("junk"))
        self.assertIsNone(self.store.get(item["id"]))
        self.assertIsNone(self.store.get("junk"))

    def test_delete_chat(self):
        for chat in ("a", "a", "b"):
            self.add("q", chat=chat)
        self.assertEqual(self.store.delete_chat("a"), 2)
        self.assertEqual(self.store.delete_chat("a"), 0)
        self.assertEqual(self.store.stats()["total"], 1)


class ListTests(StoreCase):
    def setUp(self):
        super().setUp()
        self.a = self.add("Alpha question", "alpha answer")
        self.b = self.add("Beta question", "beta answer")
        self.c = self.add("Gamma question", "gamma answer")
        self.store.rate(self.a["id"], 1)
        self.store.rate(self.b["id"], -1, "wrong")
        self.store.correct(self.b["id"], "fixed beta")
        self.store.set_teacher(self.c["id"], "teacher gamma", "big")

    def ids(self, **kw):
        return [i["id"] for i in self.store.list(**kw)["items"]]

    def test_newest_first_and_total(self):
        res = self.store.list()
        self.assertEqual([i["id"] for i in res["items"]], [self.c["id"], self.b["id"], self.a["id"]])
        self.assertEqual(res["total"], 3)
        self.assertEqual(self.store.list(limit=1, offset=1)["items"][0]["id"], self.b["id"])
        self.assertEqual(self.store.list(limit=1, offset=1)["total"], 3)

    def test_filters(self):
        self.assertEqual(self.ids(rating=1), [self.a["id"]])
        self.assertEqual(self.ids(rating=-1), [self.b["id"]])
        self.assertEqual(self.ids(rating=0), [self.c["id"]])
        self.assertEqual(self.ids(corrected=True), [self.b["id"]])
        self.assertEqual(self.ids(corrected=False), [self.c["id"], self.a["id"]])
        self.assertEqual(self.ids(with_teacher=True), [self.c["id"]])
        self.assertEqual(self.ids(with_teacher=False), [self.b["id"], self.a["id"]])
        self.assertEqual(self.ids(rating=-1, corrected=True, with_teacher=False), [self.b["id"]])
        self.assertEqual(self.ids(rating=1, corrected=True), [])
        with self.assertRaises(ValueError):
            self.store.list(rating=5)

    def test_search_covers_text_fields(self):
        self.assertEqual(self.ids(q="BETA"), [self.b["id"]])
        self.assertEqual(self.ids(q="fixed"), [self.b["id"]])
        self.assertEqual(self.ids(q="wrong"), [self.b["id"]])
        self.assertEqual(self.ids(q="teacher gamma"), [self.c["id"]])
        self.assertEqual(self.ids(q="  alpha  "), [self.a["id"]])
        self.assertEqual(self.ids(q="nothing like this"), [])
        self.assertEqual(len(self.ids(q="   ")), 3)

    def test_like_wildcards_are_literal(self):
        self.add("100% sure about snake_case", "x")
        self.add("one hundred", "y")
        self.assertEqual(len(self.ids(q="100%")), 1)
        self.assertEqual(len(self.ids(q="e_c")), 1)
        self.assertEqual(self.ids(q="%"), self.ids(q="100%"))
        self.assertEqual(self.ids(q="_"), self.ids(q="e_c"))
        self.assertEqual(self.ids(q="\\"), [])
        self.assertEqual(self.ids(q="'; DROP TABLE exchanges; --"), [])
        self.assertEqual(self.store.stats()["total"], 5)

    def test_absurd_numbers_do_not_raise(self):
        self.assertEqual(self.store.list(offset=10**30, limit=10**30)["items"], [])
        self.assertEqual(self.store.list(q="\x00")["total"], 3)
        self.assertEqual(self.store.list(q="al\x00pha")["total"], 1)
        self.assertIsNone(self.store.get(10**30))
        self.assertFalse(self.store.delete(-10**30))
        self.assertIsNone(self.store.rate(10**30, 1))
        self.assertEqual(len(self.store.recent_without_teacher(10**30)), 2)

    def test_limit_is_clamped(self):
        self.assertEqual(len(self.store.list(limit=0)["items"]), 1)
        self.assertEqual(len(self.store.list(limit=10_000)["items"]), 3)
        self.assertEqual(self.store.list(offset=-5)["items"][0]["id"], self.c["id"])

    def test_list_messages_only_on_request(self):
        self.assertTrue(all("messages" not in i for i in self.store.list()["items"]))
        items = self.store.list(include_messages=True)["items"]
        self.assertEqual(items[0]["messages"], convo("Gamma question"))
        self.assertTrue(all("messages_json" not in i for i in items))

    def test_stats(self):
        self.assertEqual(self.store.stats(), {"total": 3, "up": 1, "down": 1, "corrected": 1, "with_teacher": 1})
        self.assertEqual(ExchangeStore(self.open_db("empty.db")).stats(),
                         {"total": 0, "up": 0, "down": 0, "corrected": 0, "with_teacher": 0})


class ExampleTests(StoreCase):
    def test_returns_similar_liked_answers_best_first(self):
        self.liked("How do I sort a list in Python?", "Use sorted().")
        best = self.liked("Python sort list by key function", "Use sorted(xs, key=...).")
        self.liked("Best pizza dough recipe", "Mix flour and water.")
        found = self.store.examples_for("how can I sort a python list by key?", k=2)
        self.assertEqual([e["id"] for e in found][0], best["id"])
        self.assertEqual(len(found), 2)
        self.assertEqual(set(found[0]), {"id", "prompt", "answer", "score"})
        self.assertGreaterEqual(found[0]["score"], found[1]["score"])
        self.assertNotIn("pizza", " ".join(e["prompt"] for e in found).lower())

    def test_only_liked_or_corrected_candidates(self):
        self.add("sorting python lists", "unrated")
        self.disliked("sorting python lists please", "bad")
        self.assertEqual(self.store.examples_for("sorting python lists"), [])
        corrected = self.add("sorting python lists again", "meh")
        self.store.correct(corrected["id"], "the fixed answer")
        found = self.store.examples_for("sorting python lists")
        self.assertEqual([e["id"] for e in found], [corrected["id"]])

    def test_correction_beats_answer(self):
        item = self.liked("explain recursion simply", "original answer")
        self.store.correct(item["id"], "corrected answer")
        self.assertEqual(self.store.examples_for("explain recursion")[0]["answer"], "corrected answer")

    def test_corrected_but_downvoted_still_counts(self):
        item = self.add("explain recursion simply", "original")
        self.store.rate(item["id"], -1, "wrong")
        self.store.correct(item["id"], "better")
        self.assertEqual(self.store.examples_for("explain recursion")[0]["answer"], "better")

    def test_minimum_overlap(self):
        self.liked("Best pizza dough recipe", "Mix flour and water.")
        self.assertEqual(self.store.examples_for("What is the capital of France?"), [])
        self.assertEqual(len(self.store.examples_for("pizza")), 1)
        self.assertEqual(len(self.store.examples_for("pizza dough")), 1)
        # one shared word out of many query words is not enough
        self.assertEqual(self.store.examples_for("tell me about the history of pizza in medieval europe"), [])
        self.assertEqual(len(self.store.examples_for("how is pizza dough made")), 1)

    def test_empty_and_stopword_queries_return_nothing(self):
        self.liked("Best pizza dough recipe", "Mix flour and water.")
        for q in ("", "   ", "the a of", "?!", "x", "🍕"):
            self.assertEqual(self.store.examples_for(q), [], repr(q))

    def test_plural_forms_match(self):
        self.liked("list the Python libraries", "Here they are.")
        self.assertEqual(len(self.store.examples_for("a python library")), 1)

    def test_unicode_prompts(self):
        self.liked("Wie funktioniert die Müllabfuhr in Zürich?", "Dienstags.")
        self.assertEqual(len(self.store.examples_for("müllabfuhr zürich")), 1)

    def test_duplicates_collapse_to_newest(self):
        self.liked("Explain   Recursion", "old answer")
        newest = self.liked("explain recursion", "new answer")
        found = self.store.examples_for("explain recursion")
        self.assertEqual([e["id"] for e in found], [newest["id"]])

    def test_budget_skips_oversized_examples(self):
        self.liked("explain recursion", "x" * 2000)
        small = self.liked("explain recursion in detail", "short")
        found = self.store.examples_for("explain recursion", k=2, max_chars=1500)
        self.assertEqual([e["id"] for e in found], [small["id"]])
        for e in self.store.examples_for("explain recursion", k=2, max_chars=3000):
            self.assertLessEqual(len(e["prompt"]) + len(e["answer"]), 1500)
        self.assertEqual(len(self.store.examples_for("explain recursion", k=1, max_chars=5000)), 1)
        self.assertEqual(len(self.store.examples_for("explain recursion", k=2, max_chars=5000)), 2)

    def test_k_and_budget_edge_cases(self):
        self.liked("explain recursion", "answer")
        self.assertEqual(self.store.examples_for("explain recursion", k=0), [])
        self.assertEqual(self.store.examples_for("explain recursion", k=-1), [])
        self.assertEqual(self.store.examples_for("explain recursion", max_chars=0), [])
        self.assertEqual(self.store.examples_for("explain recursion", k=5, max_chars=3), [])

    def test_content_as_list_prompt_is_matched(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "explain recursion simply"}]}]
        item = self.store.record("c", 0, msgs, "Think of mirrors.")
        self.store.rate(item["id"], 1)
        self.assertEqual(self.store.examples_for("explain recursion")[0]["answer"], "Think of mirrors.")

    def test_only_newest_candidates_are_scanned(self):
        old = self.liked("explain recursion", "old")
        for i in range(3):
            self.liked(f"unrelated gardening topic {i}", "x")
        with mock.patch.object(feedback, "MAX_SCAN", 3):
            self.assertEqual(self.store.examples_for("explain recursion"), [])
        self.assertEqual(self.store.examples_for("explain recursion")[0]["id"], old["id"])

    def test_huge_prompts_and_queries(self):
        self.liked("explain recursion " + "filler " * 20_000, "answer")
        self.assertEqual(self.store.examples_for("explain recursion", max_chars=10**7)[0]["answer"], "answer")
        self.assertEqual(len(self.store.examples_for("explain recursion " + "zzz " * 100_000, max_chars=10**7)), 1)

    def test_hostile_queries_never_raise(self):
        self.liked("explain recursion", "answer")
        for q in ('"', "*", "OR", "NEAR(", ")", "-", "'; DROP TABLE exchanges; --", "\x00", "\\", "%", "🙂" * 50):
            self.store.examples_for(q)
            self.store.lessons_for(q)
            self.store.build_block(q, 2, 500)


class LessonTests(StoreCase):
    def test_lessons_for_similar_prompts(self):
        mine = self.disliked("How do I reverse a list in Python?", "Too long, show a one-liner")
        self.disliked("Best pizza dough recipe", "Skip the history")
        self.liked("How do I reverse a string in Python?", "s[::-1]")
        found = self.store.lessons_for("reverse a list python")
        self.assertEqual([lesson["id"] for lesson in found], [mine["id"]])
        self.assertEqual(set(found[0]), {"id", "prompt", "note", "score"})
        self.assertEqual(found[0]["note"], "Too long, show a one-liner")

    def test_needs_downvote_and_note(self):
        self.disliked("reverse a list python", "")
        silent = self.add("reverse a list python")
        self.store.rate(silent["id"], 0, "note without a vote")
        self.assertEqual(self.store.lessons_for("reverse a list python"), [])

    def test_ranked_limited_and_deduplicated(self):
        for i in range(5):
            self.disliked(f"reverse a list python variant{i}", f"note {i}")
        self.disliked("reverse a list python variant0", "note 0")
        found = self.store.lessons_for("reverse a list python variant0", k=3)
        self.assertEqual(len(found), 3)
        self.assertEqual(found[0]["note"], "note 0")
        self.assertEqual(len({(lesson["prompt"], lesson["note"]) for lesson in found}), 3)
        self.assertEqual(self.store.lessons_for("reverse a list python", k=0), [])
        self.assertEqual(self.store.lessons_for("", k=3), [])

    def test_unrelated_prompts_are_not_returned(self):
        self.disliked("reverse a list python", "short please")
        self.assertEqual(self.store.lessons_for("what is the capital of France"), [])


class BuildBlockTests(StoreCase):
    def setUp(self):
        super().setUp()
        self.ex1 = self.liked("How do I sort a list in Python?", "Use sorted(xs).")
        self.ex2 = self.liked("Sort a python list in reverse order", "sorted(xs, reverse=True)")
        self.les = self.disliked("Sort python list by length", "Don't lecture, give the code")

    def test_block_format(self):
        block = self.store.build_block("sort a python list", 2, 4000)
        text = block["text"]
        self.assertTrue(text.startswith(feedback.EXAMPLES_HEAD + "\n\nQ: "))
        self.assertIn("\nA: Use sorted(xs).", text)
        self.assertIn("\n\nReminders from past feedback:\n- When asked something like "
                      "“Sort python list by length”: Don't lecture, give the code", text)
        self.assertEqual(sorted(block["example_ids"]), sorted([self.ex1["id"], self.ex2["id"]]))
        self.assertEqual(block["lesson_ids"], [self.les["id"]])
        self.assertLess(text.index("Q: "), text.index(feedback.REMINDERS_HEAD))

    def test_lessons_flag_and_k_zero(self):
        without = self.store.build_block("sort a python list", 2, 4000, lessons=False)
        self.assertNotIn("Reminders", without["text"])
        self.assertEqual(without["lesson_ids"], [])
        only = self.store.build_block("sort a python list", 0, 4000)
        self.assertTrue(only["text"].startswith(feedback.REMINDERS_HEAD))
        self.assertNotIn("Examples of answers", only["text"])
        self.assertEqual(only["example_ids"], [])

    def test_nothing_relevant_gives_empty_block(self):
        self.assertEqual(self.store.build_block("capital of France", 2, 4000),
                         {"text": "", "example_ids": [], "lesson_ids": []})
        self.assertEqual(self.store.build_block("", 2, 4000)["text"], "")

    def test_max_chars_is_strict(self):
        self.store.correct(self.ex2["id"], "y" * 400)
        for k in (1, 2, 3):
            for max_chars in list(range(-5, 400, 7)) + [600, 800, 1200, 4000]:
                text = self.store.build_block("sort a python list", k, max_chars)["text"]
                self.assertLessEqual(len(text), max(0, max_chars), (k, max_chars))
        generous = self.store.build_block("sort a python list", 2, 4000)
        self.assertEqual(len(generous["example_ids"]), 2)

    def test_reminders_use_room_the_examples_left(self):
        self.store.delete(self.ex1["id"])
        self.store.delete(self.ex2["id"])
        long_note = "Please " + "keep it short and sweet " * 12
        item = self.add("Sort python list by price")
        self.store.rate(item["id"], -1, long_note)
        text = self.store.build_block("sort python list by price", 2, 600)["text"]
        self.assertNotIn("Examples of answers", text)
        self.assertGreater(len(text), 600 // 3)
        self.assertLessEqual(len(text), 600)

    def test_returned_ids_match_text(self):
        block = self.store.build_block("sort a python list", 2, 330)
        self.assertEqual(len(block["example_ids"]), 1)
        candidates = self.store.examples_for("sort a python list", 2, 4000)
        shown = {e["id"] for e in candidates if e["answer"] in block["text"]}
        self.assertEqual(shown, set(block["example_ids"]))

    def test_long_lesson_is_shortened_not_dropped(self):
        long_note = "never " + "ramble " * 200
        item = self.add("Sort python list by weight")
        self.store.rate(item["id"], -1, long_note)
        block = self.store.build_block("sort python list by weight", 0, 1000)
        line = [ln for ln in block["text"].splitlines() if ln.startswith("- ")][0]
        self.assertLessEqual(len(line), 300)
        self.assertTrue(line.endswith("…"))
        self.assertIn(item["id"], block["lesson_ids"])

    def test_multiline_notes_stay_on_one_line(self):
        item = self.add("Sort python list by colour")
        self.store.rate(item["id"], -1, "line one\nline two\n\n- sneaky bullet")
        block = self.store.build_block("sort python list by colour", 0, 1000)
        lines = [ln for ln in block["text"].splitlines() if "line one" in ln]
        self.assertEqual(len(lines), 1)
        self.assertIn("line two - sneaky bullet", lines[0])
        self.assertTrue(lines[0].startswith("- "))


class RecentTests(StoreCase):
    def test_recent_without_teacher(self):
        a = self.add("first")
        b = self.add("second")
        c = self.add("third")
        self.store.set_teacher(b["id"], "teacher")
        self.store.rate(c["id"], -1, "bad")
        self.store.record("c9", 0, [{"role": "system", "content": "s"}], "no prompt")
        found = self.store.recent_without_teacher(10)
        self.assertEqual([i["id"] for i in found], [c["id"], a["id"]])
        self.assertEqual(found[0]["messages"], convo("third"))
        self.assertIs(found[0]["used_tools"], False)
        self.assertEqual([i["id"] for i in self.store.recent_without_teacher(1)], [c["id"]])
        self.assertEqual(self.store.recent_without_teacher(0), [])

    def test_since_filter(self):
        old = self.add("old")
        self.db.execute("UPDATE exchanges SET created = ? WHERE id = ?", (old["created"] - 100, old["id"]))
        new = self.add("new")
        cutoff = new["created"] - 50
        self.assertEqual([i["id"] for i in self.store.recent_without_teacher(10, since=cutoff)], [new["id"]])
        self.assertEqual(len(self.store.recent_without_teacher(10)), 2)


class HelperTests(unittest.TestCase):
    def test_tokens(self):
        self.assertEqual(feedback._tokens("How do I sort the Libraries and boxes, 3D?"),
                         ["sort", "library", "boxe", "3d"])
        self.assertEqual(feedback._tokens("The class and this bus"), ["class", "bus"])
        self.assertEqual(feedback._tokens("🍕 !! _ -"), [])

    def test_fit(self):
        self.assertEqual(feedback._fit("hello", 10), "hello")
        self.assertEqual(feedback._fit("hello world", 6), "hello…")
        self.assertEqual(feedback._fit("hello", 0), "")

    def test_rank_orders_by_relevance_and_keeps_ties_stable(self):
        texts = ["python lists", "python python lists sorting", "gardening", "python lists"]
        ranked = feedback._rank("python sorting lists", texts)
        self.assertEqual([i for _, i in ranked][0], 1)
        self.assertEqual([i for _, i in ranked][1:], [0, 3])
        self.assertEqual(feedback._rank("python", []), [])


if __name__ == "__main__":
    unittest.main()


class PurgeAndRetention(unittest.TestCase):
    """Review finding: deleting a chat in the UI left its transcripts, queued jobs and training examples behind."""

    def setUp(self):
        import tempfile
        from duckstoken.db import Database
        from duckstoken.memory import MemoryStore
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Database(Path(self.tmp.name) / "p.db")
        self.addCleanup(self.db.close)
        self.store = ExchangeStore(self.db, MemoryStore(self.db))

    def add(self, chat, idx, **kw):
        return self.store.record(chat, idx, [{"role": "user", "content": f"q{idx}"}], f"a{idx}", **kw)

    def test_purge_chat_removes_log_queue_and_training_examples_of_that_chat_only(self):
        a1, a2, b1 = self.add("A", 1), self.add("A", 3), self.add("B", 1)
        now = __import__("time").time()
        for ex in (a1, b1):
            self.db.execute("INSERT INTO dataset_items(exchange_id, kind, messages_json, chosen, source, created) VALUES (?, 'sft', '[]', 'x', 'liked', ?)", (ex["id"], now))
        self.db.execute("INSERT INTO queue(created, chat_id, exchange_id) VALUES (?, 'A', ?)", (now, a2["id"]))
        out = self.store.purge_chat("A")
        self.assertEqual(out, {"exchanges": 2, "dataset_items": 1, "queue": 1})
        self.assertEqual(self.db.scalar("SELECT count(*) FROM exchanges"), 1)       # chat B untouched
        self.assertEqual(self.db.scalar("SELECT count(*) FROM dataset_items"), 1)
        self.assertEqual(self.store.purge_chat("A"), {"exchanges": 0, "dataset_items": 0, "queue": 0})

    def test_retention_keeps_what_you_rated_corrected_or_approved(self):
        old = __import__("time").time() - 100 * 86400
        plain, rated, corrected, approved = (self.add("R", i) for i in (1, 2, 3, 4))
        self.store.rate(rated["id"], 1)
        self.store.correct(corrected["id"], "better answer")
        self.db.execute("INSERT INTO dataset_items(exchange_id, kind, messages_json, chosen, source, created) VALUES (?, 'sft', '[]', 'x', 'liked', ?)", (approved["id"], old))
        self.db.execute("UPDATE exchanges SET created = ?", (old,))
        self.assertEqual(self.store.prune(0), 0)            # 0 = keep everything
        self.assertEqual(self.store.prune(30), 1)           # only the plain, unrated one
        left = {r["id"] for r in self.db.query("SELECT id FROM exchanges")}
        self.assertEqual(left, {rated["id"], corrected["id"], approved["id"]})
        self.assertEqual(self.store.prune(30), 0)
