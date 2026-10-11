import datetime
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

os.environ["DUCKSTOKEN_HOME"] = tempfile.mkdtemp(prefix="duck-escalation-")

from gguf_helper import write_gguf  # noqa: E402
from duckstoken import escalation, llm  # noqa: E402
from duckstoken.db import Database  # noqa: E402
from duckstoken.escalation import (EscalationWorker, QueueFull, QueueStore, big_settings, hard_prompt,  # noqa: E402
                                   in_window, unsure_answer)

BIG_PATH = "/models/gpt-oss-120b-mxfp4.gguf"


def convo(question="What is 2+2?", *earlier):
    return [*earlier, {"role": "user", "content": question}]


class TempDbCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Database(Path(self.tmp.name) / "x.db")
        self.addCleanup(self.db.close)  # Windows cannot delete a database that is still open
        self.queue = QueueStore(self.db)


# ------------------------------------------------------------------------------------------------ the queue
class QueueTests(TempDbCase):
    def test_enqueue_stores_a_queued_job(self):
        job = self.queue.enqueue(convo("Why is the sky blue?"), chat_id="c1", exchange_id=4, reason="downvote",
                                 priority=3)
        self.assertEqual((job["status"], job["reason"], job["chat_id"], job["exchange_id"], job["priority"]),
                         ("queued", "downvote", "c1", 4, 3))
        self.assertEqual(json.loads(job["messages_json"]), convo("Why is the sky blue?"))
        self.assertEqual((job["answer"], job["error"], job["started"], job["finished"]), ("", "", 0, 0))
        self.assertEqual(self.queue.get(job["id"]), job)

    def test_enqueue_requires_a_user_message(self):
        for bad in ([], [{"role": "assistant", "content": "hi"}], [{"role": "user", "content": "   "}],
                    [{"role": "user", "content": None}], "not a list", None, [{"content": "no role"}, 5, None]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.queue.enqueue(bad)
        self.assertEqual(self.queue.counts()["queued"], 0)

    def test_enqueue_validates_reason_and_exchange_id(self):
        with self.assertRaises(ValueError):
            self.queue.enqueue(convo(), reason="because")
        for bad in ("abc", True, 1.5e30, -3):
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.queue.enqueue(convo(), exchange_id=bad)
        self.assertEqual(self.queue.enqueue(convo(), exchange_id="12")["exchange_id"], 12)

    def test_priority_is_clamped(self):
        self.assertEqual(self.queue.enqueue(convo("a"), priority=10_000)["priority"], 100)
        self.assertEqual(self.queue.enqueue(convo("b"), priority=-10_000)["priority"], -100)

    def test_content_parts_become_text_and_junk_is_dropped(self):
        parts = [{"type": "text", "text": "Describe"}, {"type": "image_url", "image_url": {"url": "data:..."}},
                 {"type": "text", "text": "this"}]
        job = self.queue.enqueue([{"role": "system", "content": "Be brief.", "reasoning": "secret"},
                                  {"role": "weird", "content": "dropped"}, "junk",
                                  {"role": "user", "content": parts}])
        self.assertEqual(json.loads(job["messages_json"]), [{"role": "system", "content": "Be brief."},
                                                           {"role": "user", "content": "Describe\nthis"}])

    def test_tool_plumbing_is_kept(self):
        calls = [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]
        job = self.queue.enqueue([{"role": "user", "content": "Read it"},
                                  {"role": "assistant", "content": "", "tool_calls": calls},
                                  {"role": "tool", "content": "file text", "tool_call_id": "c1", "extra": 1}])
        self.assertEqual(json.loads(job["messages_json"])[1:], [
            {"role": "assistant", "content": "", "tool_calls": calls},
            {"role": "tool", "content": "file text", "tool_call_id": "c1"}])

    def test_oversized_conversations_lose_their_oldest_messages_first(self):
        chunk = "x" * 150_000
        messages = [{"role": "user", "content": "old " + chunk}, {"role": "assistant", "content": chunk},
                    {"role": "user", "content": "middle " + chunk}, {"role": "assistant", "content": chunk},
                    {"role": "user", "content": "the real question"}]
        kept = json.loads(self.queue.enqueue(messages)["messages_json"])
        self.assertEqual(kept[-1]["content"], "the real question")
        self.assertLess(len(kept), len(messages))
        self.assertLessEqual(sum(len(m["content"]) for m in kept), escalation.MAX_JOB_CHARS)
        self.assertNotIn("old", kept[0]["content"][:4])

    def test_a_dropped_start_never_leaves_an_orphan_tool_result(self):
        big = "y" * 200_000
        messages = [{"role": "assistant", "content": big, "tool_calls": [{"id": "a"}]},
                    {"role": "tool", "content": "result", "tool_call_id": "a"},
                    {"role": "user", "content": big}, {"role": "user", "content": "follow-up"}]
        kept = json.loads(self.queue.enqueue(messages)["messages_json"])
        self.assertEqual([m["role"] for m in kept], ["user", "user"])

    def test_a_conversation_that_cannot_be_shortened_enough_is_rejected(self):
        messages = [{"role": "user", "content": "q" * 250_000}, {"role": "tool", "content": "t" * 250_000,
                                                                "tool_call_id": "k"}]
        with self.assertRaises(ValueError):
            self.queue.enqueue(messages)
        self.assertEqual(self.queue.counts()["queued"], 0)

    def test_queue_limit_counts_only_waiting_jobs(self):
        first = self.queue.enqueue(convo("a"), max_queued=2)
        self.queue.enqueue(convo("b"), max_queued=2)
        with self.assertRaises(QueueFull) as ctx:
            self.queue.enqueue(convo("c"), max_queued=2)
        self.assertIsInstance(ctx.exception, ValueError)
        self.queue.mark_running(first["id"])  # a running job no longer waits
        self.assertEqual(self.queue.enqueue(convo("c"), max_queued=2)["status"], "queued")
        self.assertEqual(self.queue.counts()["queued"], 2)

    def test_identical_queued_job_for_the_same_exchange_is_deduplicated(self):
        a = self.queue.enqueue(convo("q"), exchange_id=9)
        self.assertEqual(self.queue.enqueue(convo("q"), exchange_id=9)["id"], a["id"])
        self.assertNotEqual(self.queue.enqueue(convo("q"), exchange_id=10)["id"], a["id"])
        self.assertNotEqual(self.queue.enqueue(convo("changed"), exchange_id=9)["id"], a["id"])
        self.assertEqual(self.queue.counts()["queued"], 3)

    def test_no_dedupe_without_an_exchange(self):
        self.assertNotEqual(self.queue.enqueue(convo("q"))["id"], self.queue.enqueue(convo("q"))["id"])

    def test_dedupe_only_applies_while_queued(self):
        a = self.queue.enqueue(convo("q"), exchange_id=9)
        self.queue.mark_running(a["id"])
        self.assertNotEqual(self.queue.enqueue(convo("q"), exchange_id=9)["id"], a["id"])

    def test_dedupe_wins_over_the_limit(self):
        a = self.queue.enqueue(convo("q"), exchange_id=9, max_queued=1)
        self.assertEqual(self.queue.enqueue(convo("q"), exchange_id=9, max_queued=1)["id"], a["id"])

    def test_next_queued_orders_by_priority_then_age(self):
        low = self.queue.enqueue(convo("low"), priority=0)
        high1 = self.queue.enqueue(convo("high1"), priority=5)
        high2 = self.queue.enqueue(convo("high2"), priority=5)
        review = self.queue.enqueue(convo("review"), priority=-1)
        order = []
        while (job := self.queue.next_queued()) is not None:
            order.append(job["id"])
            self.queue.mark_running(job["id"])
        self.assertEqual(order, [high1["id"], high2["id"], low["id"], review["id"]])

    def test_lifecycle(self):
        job = self.queue.enqueue(convo(), exchange_id=1)
        self.assertTrue(self.queue.mark_running(job["id"]))
        self.assertFalse(self.queue.mark_running(job["id"]))
        running = self.queue.get(job["id"])
        self.assertEqual(running["status"], "running")
        self.assertGreater(running["started"], 0)
        self.queue.finish(job["id"], "four", tokens=12, tps=3.5, model="gpt-oss-120b")
        done = self.queue.get(job["id"])
        self.assertEqual((done["status"], done["answer"], done["tokens"], done["tps"], done["model"]),
                         ("done", "four", 12, 3.5, "gpt-oss-120b"))
        self.assertGreaterEqual(done["finished"], done["started"])

    def test_fail_keeps_a_bounded_error(self):
        job = self.queue.enqueue(convo())
        self.queue.mark_running(job["id"])
        self.queue.fail(job["id"], "boom " * 2000)
        failed = self.queue.get(job["id"])
        self.assertEqual(failed["status"], "error")
        self.assertEqual(len(failed["error"]), escalation.MAX_ERROR)

    def test_cancel_only_affects_queued_jobs(self):
        queued, running = self.queue.enqueue(convo("a")), self.queue.enqueue(convo("b"))
        self.queue.mark_running(running["id"])
        self.assertTrue(self.queue.cancel(queued["id"]))
        self.assertFalse(self.queue.cancel(queued["id"]))
        self.assertFalse(self.queue.cancel(running["id"]))
        self.assertFalse(self.queue.cancel("nope"))
        self.assertFalse(self.queue.cancel(99999))
        self.assertEqual(self.queue.get(queued["id"])["status"], "cancelled")
        self.assertEqual(self.queue.get(running["id"])["status"], "running")
        self.assertFalse(self.queue.mark_running(queued["id"]))

    def test_clear_finished_keeps_active_jobs(self):
        ids = [self.queue.enqueue(convo(str(i)))["id"] for i in range(5)]
        self.queue.mark_running(ids[0])
        self.queue.mark_running(ids[1])
        self.queue.finish(ids[1], "ok")
        self.queue.mark_running(ids[2])
        self.queue.fail(ids[2], "bad")
        self.queue.cancel(ids[3])
        self.assertEqual(self.queue.clear_finished(), 3)
        self.assertEqual(sorted(j["id"] for j in self.queue.list()), sorted([ids[0], ids[4]]))

    def test_counts_cover_every_status(self):
        self.assertEqual(self.queue.counts(), {"queued": 0, "running": 0, "done": 0, "error": 0, "cancelled": 0})
        a, b = self.queue.enqueue(convo("a")), self.queue.enqueue(convo("b"))
        self.queue.mark_running(a["id"])
        self.assertEqual(self.queue.counts(), {"queued": 1, "running": 1, "done": 0, "error": 0, "cancelled": 0})
        self.queue.cancel(b["id"])
        self.assertEqual(self.queue.counts()["cancelled"], 1)

    def test_list_is_newest_first_and_filterable(self):
        ids = [self.queue.enqueue(convo(str(i)))["id"] for i in range(4)]
        self.queue.mark_running(ids[0])
        self.assertEqual([j["id"] for j in self.queue.list()], ids[::-1])
        self.assertEqual([j["id"] for j in self.queue.list(status="running")], [ids[0]])
        self.assertEqual(len(self.queue.list(limit=2)), 2)
        self.assertEqual(len(self.queue.list(limit=0)), 1)  # limits are clamped, never negative or unbounded
        self.assertEqual(self.queue.list(status="x' OR '1'='1"), [])
        self.assertIn("messages_json", self.queue.list()[0])
        self.assertNotIn("messages_json", self.queue.list(include_messages=False)[0])

    def test_hostile_text_is_stored_verbatim(self):
        evil = "x'); DROP TABLE queue;--"
        job = self.queue.enqueue(convo(evil), chat_id=evil)
        self.assertEqual(self.queue.get(job["id"])["chat_id"], evil)
        self.assertEqual(self.queue.counts()["queued"], 1)

    def test_long_chat_ids_are_truncated(self):
        self.assertEqual(len(self.queue.enqueue(convo(), chat_id="c" * 5000)["chat_id"]), escalation.MAX_NAME)

    def test_get_with_a_bad_id(self):
        for bad in (None, "x", -1, 10**30, True, 4.5e40):
            self.assertIsNone(self.queue.get(bad))

    def test_requeue_running(self):
        a, b, c = (self.queue.enqueue(convo(str(i))) for i in range(3))
        self.queue.mark_running(a["id"])
        self.queue.mark_running(b["id"])
        self.assertEqual(self.queue.requeue_running(a["id"]), 1)
        self.assertEqual(self.queue.get(a["id"])["status"], "queued")
        self.assertEqual(self.queue.get(a["id"])["started"], 0)
        self.assertEqual(self.queue.requeue_running(), 1)
        self.assertEqual(self.queue.get(b["id"])["status"], "queued")
        self.assertEqual(self.queue.requeue_running(), 0)
        self.assertEqual(self.queue.get(c["id"])["status"], "queued")


# ------------------------------------------------------------------------------------------- night window
class InWindowTests(unittest.TestCase):
    def t(self, text):
        h, m = text.split(":")
        return datetime.time(int(h), int(m))

    def test_plain_window(self):
        for now, expected in (("00:59", False), ("01:00", True), ("03:30", True), ("06:59", True), ("07:00", False),
                              ("12:00", False), ("23:59", False)):
            self.assertEqual(in_window(self.t(now), "01:00", "07:00"), expected, now)

    def test_window_wrapping_past_midnight(self):
        for now, expected in (("21:59", False), ("22:00", True), ("23:59", True), ("00:00", True), ("05:59", True),
                              ("06:00", False), ("12:00", False)):
            self.assertEqual(in_window(self.t(now), "22:00", "06:00"), expected, now)

    def test_seconds_do_not_matter_and_datetimes_work(self):
        self.assertTrue(in_window(datetime.time(6, 59, 59), "01:00", "07:00"))
        self.assertFalse(in_window(datetime.time(7, 0, 0), "01:00", "07:00"))
        self.assertTrue(in_window(datetime.datetime(2025, 1, 1, 2, 0), "01:00", "07:00"))
        self.assertTrue(in_window(datetime.time(2, 0, tzinfo=datetime.timezone.utc), "01:00", "07:00"))

    def test_loose_formatting_is_accepted(self):
        self.assertTrue(in_window(self.t("02:00"), " 1:00 ", "7:00"))

    def test_invalid_input_is_never_in_the_window(self):
        for start, end in (("", ""), ("abc", "07:00"), ("01:00", "25:00"), ("24:00", "07:00"), ("01:60", "07:00"),
                           ("1:5", "07:00"), ("01:00", None), (None, None), (100, 200), ("01:00", "07:00:00"),
                           ("-1:00", "07:00"), ("01:00", "07:00x"), ("0１:00", "07:00")):
            self.assertFalse(in_window(self.t("02:00"), start, end), (start, end))

    def test_empty_window_is_never_active(self):
        for now in ("00:00", "07:00", "12:00"):
            self.assertFalse(in_window(self.t(now), "07:00", "07:00"))


# ---------------------------------------------------------------------------------------- server settings
class BigSettingsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.base = {"llama_server_path": "/bin/llama-server", "model_path": "/models/small.gguf", "host": "0.0.0.0",
                     "port": 8080, "api_key": "secret", "parallel": 4, "mlock": True, "ctx_size": 32768,
                     "n_cpu_moe": 24, "reasoning_effort": "low", "lora_path": "/x/lora.gguf", "lora_scale": 0.5,
                     "threads": 6, "big_model_path": BIG_PATH, "big_port": 8099, "big_ctx_size": 6000,
                     "big_n_cpu_moe": 0, "big_reasoning_effort": "high"}

    def moe_model(self, layers=36, experts=128):
        path = self.dir / "big.gguf"
        write_gguf(path, {"general.architecture": "gpt-oss", "gpt-oss.block_count": layers,
                          "gpt-oss.expert_count": experts, "gpt-oss.expert_used_count": 4})
        return str(path)

    def test_big_values_replace_the_everyday_ones(self):
        out = big_settings(self.base)
        self.assertEqual((out["model_path"], out["port"], out["ctx_size"], out["api_key"], out["host"]),
                         (BIG_PATH, 8099, 6000, "", "127.0.0.1"))
        self.assertEqual((out["parallel"], out["mlock"], out["reasoning_effort"], out["lora_path"]),
                         (1, False, "high", ""))

    def test_everything_else_is_carried_over_and_the_input_is_untouched(self):
        before = dict(self.base)
        out = big_settings(self.base)
        self.assertEqual((out["llama_server_path"], out["threads"], out["lora_scale"]), ("/bin/llama-server", 6, 0.5))
        self.assertEqual(self.base, before)
        self.assertIsNot(out, self.base)

    def test_explicit_cpu_moe_layers_win(self):
        self.base.update(big_n_cpu_moe=20, big_model_path=self.moe_model())
        self.assertEqual(big_settings(self.base)["n_cpu_moe"], 20)

    def test_moe_models_keep_all_experts_on_the_cpu_by_default(self):
        self.base["big_model_path"] = self.moe_model(layers=36)
        self.assertEqual(big_settings(self.base)["n_cpu_moe"], 36)

    def test_dense_models_get_no_cpu_moe(self):
        self.base["big_model_path"] = self.moe_model(experts=0)
        self.assertEqual(big_settings(self.base)["n_cpu_moe"], 0)

    def test_unreadable_models_fall_back_to_zero(self):
        broken = self.dir / "broken.gguf"
        broken.write_bytes(b"not a gguf at all")
        for path in ("", str(self.dir / "missing.gguf"), str(broken), str(self.dir)):
            self.base["big_model_path"] = path
            self.assertEqual(big_settings(self.base)["n_cpu_moe"], 0, path)

    def test_missing_keys_use_defaults(self):
        out = big_settings({})
        self.assertEqual((out["port"], out["ctx_size"], out["n_cpu_moe"], out["model_path"]), (8081, 4096, 0, ""))


# ---------------------------------------------------------------------------------------------- heuristics
class HardPromptTests(unittest.TestCase):
    POSITIVE = [
        "Prove that the square root of 2 is irrational.",
        "Derive the closed form for the sum of the first n squares and explain each step.",
        "Design a system architecture for a real-time chat app with millions of users",
        "Compare the trade-offs between PostgreSQL and MongoDB for an analytics workload.",
        "What is the time complexity of quicksort in the worst case and why?",
        "My program crashes with a segmentation fault when I free the buffer, what is the root cause?",
        "Optimize this query for a table with 10 million rows please",
        "Why do I get this?\nTraceback (most recent call last):\n  File \"a.py\", line 3, in <module>\n    foo()",
        "Here is my code:\n```js\nlet a = [];\nconsole.log(a[0].x)\n```\nIt throws an error when the list is empty, "
        "can you debug it?",
        "Can you solve this equation for me: 3*x^2 + 4*x - 7 = 0 and walk me through it in detail?",
        "Please help with the following:\n1. How do I set up the database?\n2. How do I migrate the old data?\n"
        "3. How do I back it up nightly?",
        "I have a long question. " + "It keeps going with details about my setup and what I tried. " * 10 +
        "Why does it fail with an error every time?",
    ]
    NEGATIVE = [
        "hi", "Hello there, how are you today?", "What's the capital of France?", "Write a haiku about autumn",
        "Translate 'good morning' to Spanish", "Write a Python function that reverses a string",
        "Tell me a joke about ducks", "Summarize this paragraph for me please",
        "How do I prove my identity to the bank?",
        "Can you proofread my email to the landlord?", "What is a proof of concept in a startup?",
        "What's the optimal temperature for brewing green tea?", "What time is it in Tokyo right now?",
        "How do I improve my sleep?", "Give me three ideas for a birthday party for a ten year old",
        "Is 3/4 bigger than 2/3?", "", "   ", "why?", "ok thanks",
    ]

    def test_hard_prompts(self):
        for text in self.POSITIVE:
            self.assertTrue(hard_prompt(text), text[:60])

    def test_easy_prompts(self):
        for text in self.NEGATIVE:
            self.assertFalse(hard_prompt(text), text[:60])

    def test_conversations_judge_the_last_user_message(self):
        hard = "Prove that there are infinitely many primes."
        messages = [{"role": "user", "content": hard}, {"role": "assistant", "content": "Sure..."},
                    {"role": "user", "content": "Thanks, that was nice and clear!"}]
        self.assertFalse(hard_prompt(messages))
        self.assertTrue(hard_prompt(messages[:-1] + [{"role": "user", "content": hard}]))
        self.assertTrue(hard_prompt([{"role": "user", "content": [{"type": "text", "text": hard}]}]))

    def test_odd_input_never_raises(self):
        for value in (None, 5, [], [None, "x", {"role": "user"}], [{"role": "user", "content": None}], {"a": 1}):
            self.assertFalse(hard_prompt(value))

    def test_huge_input_is_fast(self):
        started = time.time()
        hard_prompt("design the " + "very " * 200_000 + "architecture")
        hard_prompt("a? " * 100_000)
        self.assertLess(time.time() - started, 2)


class UnsureAnswerTests(unittest.TestCase):
    UNSURE = [
        "I'm not sure, but I think it might be 1969.", "I don't know the answer to that.",
        "I do not know.", "Honestly I have no idea what that error means.", "I cannot verify that claim.",
        "I can't confirm the release date.", "As an AI language model, I can't browse the internet.",
        "As an AI, I don't have access to real-time data.", "I\u2019m not sure about that one.",
        "Sorry, I can't help with that.", "I\u2019m sorry, but I cannot do that.", "I can't.", "Unable to answer.",
        "I am unable to help.", "I could be wrong here, but the answer is probably 42.",
        "It was probably in the nineties. I might be wrong though.",
        "  I'm   NOT\nsure  ",
        "Here is a long and helpful explanation. " * 60 + "Anyway, I'm not sure that is right.",
        "I don't know. " + "Details follow here. " * 100,
    ]
    CONFIDENT = [
        "The capital of France is Paris.", "", None, "   ", "Sure! Here is the code:\n```python\nprint('hi')\n```",
        "You don't know what you are missing, this recipe is great.",
        "Sorry for the confusion earlier, the answer is 42.",
        "I can't wait to show you how this works! " + "Here is the full explanation of the approach. " * 5,
        "A long, careful answer that explains every step of the method in detail. " * 60 +
        "In the middle there is a quote: 'I don't know' said the character. " + "More explanation here. " * 60,
        "As an AI image generator pioneer, Midjourney was founded in 2021.",
    ]

    def test_unsure_answers(self):
        for text in self.UNSURE:
            self.assertTrue(unsure_answer(text), str(text)[:60])

    def test_confident_answers(self):
        for text in self.CONFIDENT:
            self.assertFalse(unsure_answer(text), str(text)[:60])

    def test_the_middle_of_a_very_long_answer_is_not_judged(self):
        middle = "word " * 400 + "I don't know. " + "word " * 400
        self.assertFalse(unsure_answer("Intro. " + "pad " * 300 + middle + "pad " * 300 + "Outro."))
        self.assertTrue(unsure_answer("Intro. I don't know. " + "pad " * 600))


# --------------------------------------------------------------------------------------------- the worker
class FakeServer:
    """Stands in for server_proc.ServerManager; `world.events` records what the worker does to it."""

    def __init__(self, world, name, state="stopped", load_polls=0):
        self.world, self.name, self.state, self.load_polls = world, name, state, load_polls
        self.error, self.start_error, self.die_while_loading = "", None, None
        self.starts, self.stops, self.settings, self._polls = 0, 0, [], 0
        self.on_stop = None

    def start(self, settings):
        self.world.note(self.name, "start")
        if self.start_error:
            raise RuntimeError(self.start_error)
        if self.state in ("starting", "loading", "ready"):
            raise RuntimeError("The model server is already running")
        self.starts += 1
        self.settings.append(settings)
        self.state, self._polls = "loading", self.load_polls

    def stop(self):
        self.stops += 1
        self.world.note(self.name, "stop")
        self.state = "stopped"
        if self.on_stop:
            self.on_stop()

    def status(self):
        if self.state == "loading":
            if self.die_while_loading and self._polls == 0:
                self.state, self.error = "error", self.die_while_loading
            elif self._polls <= 0:
                self.state = "ready"
            else:
                self._polls -= 1
        return {"state": self.state, "error": self.error, "running": self.state in ("starting", "loading", "ready")}

    def base_url(self):
        return "http://127.0.0.1:1"

    def upstream_headers(self):
        return {}


class FakeExchanges:
    def __init__(self):
        self.teachers, self.recent, self.fail_with = {}, [], None

    def set_teacher(self, id, answer, model=""):
        if self.fail_with:
            raise self.fail_with
        self.teachers[id] = (answer, model)
        return {"id": id}

    def recent_without_teacher(self, n, since=0.0):
        return self.recent[:n]


class World:
    """Shared state for one worker test: settings, the clock, event log and a scriptable `complete`."""

    def __init__(self):
        self.events, self.worker, self.calls, self.phases = [], None, [], {}
        self.reply = {"content": "The big answer.", "reasoning": "...", "tokens": 321, "tps": 4.5, "model": "local"}
        self.error = None
        self.inside = None  # optional callable run while the big model is "answering"
        self.now = datetime.datetime(2025, 1, 6, 12, 0)

    def note(self, who, what):
        self.events.append((who, what))
        if self.worker is not None:
            self.phases[(who, what)] = self.worker.status()["phase"]

    def complete(self, server, messages, **kw):
        self.note("big", "complete")
        self.calls.append({"server": server, "messages": messages, **kw})
        if self.inside:
            self.inside()
        if self.error:
            raise self.error
        return dict(self.reply)


SETTINGS = {
    "llama_server_path": "/bin/llama-server", "model_path": "/models/small.gguf", "port": 8080, "host": "127.0.0.1",
    "big_model_path": BIG_PATH, "big_run_mode": "manual", "big_swap_models": True, "big_port": 8081,
    "big_ctx_size": 4096, "big_n_cpu_moe": 0, "big_reasoning_effort": "high", "big_max_tokens": 3000,
    "big_idle_stop_minutes": 3, "overnight_start": "01:00", "overnight_end": "07:00", "overnight_review_count": 0,
    "queue_max": 100,
}


class WorkerCase(TempDbCase):
    def setUp(self):
        super().setUp()
        self.world = World()
        self.settings = dict(SETTINGS)
        self.small = FakeServer(self.world, "small", state="ready")
        self.big = FakeServer(self.world, "big")
        self.exchanges = FakeExchanges()
        self.sleeps = []
        self.worker = EscalationWorker(self.db, lambda: dict(self.settings), self.small, self.big, self.exchanges,
                                       self.queue, complete=self.world.complete, clock=lambda: self.world.now,
                                       sleep=self.sleeps.append)
        self.world.worker = self.worker
        self.addCleanup(self.worker.stop)

    def job(self, question="Explain monads.", **kw):
        return self.queue.enqueue(convo(question), **kw)

    def events(self):
        return list(self.world.events)


class RunOneTests(WorkerCase):
    def test_nothing_to_do(self):
        self.assertFalse(self.worker.run_one())
        self.assertEqual(self.events(), [])

    def test_happy_path_with_model_swap(self):
        job = self.job(exchange_id=7)
        self.assertTrue(self.worker.run_one())
        done = self.queue.get(job["id"])
        self.assertEqual((done["status"], done["answer"], done["tokens"], done["tps"], done["error"]),
                         ("done", "The big answer.", 321, 4.5, ""))
        self.assertEqual(done["model"], "gpt-oss-120b-mxfp4")
        self.assertEqual(self.exchanges.teachers, {7: ("The big answer.", "gpt-oss-120b-mxfp4")})
        self.assertEqual(self.events(), [("small", "stop"), ("big", "start"), ("big", "complete"), ("big", "stop"),
                                         ("small", "start")])
        self.assertEqual(self.small.state, "loading")  # restarted
        self.assertEqual(self.big.state, "stopped")
        self.assertEqual(self.worker.status()["small_was_stopped"], False)

    def test_the_big_server_gets_big_settings_and_the_small_one_the_current_settings(self):
        self.job()
        self.worker.run_one()
        started = self.big.settings[0]
        self.assertEqual((started["model_path"], started["port"], started["api_key"], started["parallel"]),
                         (BIG_PATH, 8081, "", 1))
        self.assertEqual(self.small.settings[0]["model_path"], "/models/small.gguf")
        self.assertEqual(self.small.settings[0]["port"], 8080)

    def test_complete_is_called_with_the_big_model_settings(self):
        self.settings.update(big_max_tokens=1234, big_reasoning_effort="medium")
        messages = [{"role": "system", "content": "Be nice."}, {"role": "user", "content": "Q1"},
                    {"role": "assistant", "content": "A1"}, {"role": "user", "content": "Q2"}]
        self.queue.enqueue(messages)
        self.worker.run_one()
        call = self.world.calls[0]
        self.assertIs(call["server"], self.big)
        self.assertEqual(call["messages"], messages)
        self.assertEqual((call["max_tokens"], call["temperature"], call["effort"]), (1234, 0.6, "medium"))
        self.assertGreaterEqual(call["timeout"], 1800)

    def test_without_an_exchange_only_the_job_stores_the_answer(self):
        job = self.job()
        self.worker.run_one()
        self.assertEqual(self.queue.get(job["id"])["status"], "done")
        self.assertEqual(self.exchanges.teachers, {})

    def test_model_label_prefers_the_configured_file_name(self):
        self.settings["big_model_path"] = "C:\\Users\\Bob\\models\\gpt-oss-120b-00001-of-00003.gguf"
        self.world.reply["model"] = "D:\\elsewhere\\other.gguf"
        job = self.job(exchange_id=3)
        self.worker.run_one()
        self.assertEqual(self.queue.get(job["id"])["model"], "gpt-oss-120b-00001-of-00003")
        self.assertEqual(self.exchanges.teachers[3][1], "gpt-oss-120b-00001-of-00003")
        self.assertEqual(escalation._model_label("", "/a/b/model.GGUF"), "model")
        self.assertEqual(escalation._model_label(None, ""), "big model")

    def test_phases_are_reported_while_working(self):
        self.big.load_polls = 2
        self.job()
        self.worker.run_one()
        phases = self.world.phases
        self.assertEqual(phases[("small", "stop")], "swapping")
        self.assertEqual(phases[("big", "start")], "loading_big")
        self.assertEqual(phases[("big", "complete")], "answering")
        self.assertEqual(self.worker.status()["phase"], "idle")
        self.assertEqual(len([s for s in self.sleeps if s == escalation.LOAD_POLL_SECONDS]), 2)

    def test_status_describes_the_running_job(self):
        seen = {}
        job = self.job()
        self.world.inside = lambda: seen.update(self.worker.status())
        self.worker.run_one()
        self.assertEqual((seen["phase"], seen["current_job"], seen["big_state"], seen["small_was_stopped"]),
                         ("answering", job["id"], "ready", True))
        self.assertEqual(seen["counts"]["running"], 1)
        after = self.worker.status()
        self.assertEqual((after["phase"], after["current_job"], after["counts"]["done"]), ("idle", None, 1))

    def test_only_one_job_runs_at_a_time(self):
        self.job("a")
        self.job("b")
        nested = []
        self.world.inside = lambda: nested.append(self.worker.run_one())
        self.assertTrue(self.worker.run_one())
        self.assertEqual(nested, [False])
        self.assertEqual(self.queue.counts()["queued"], 1)

    def test_a_cancelled_job_is_skipped(self):
        job = self.job()
        original = self.queue.mark_running
        self.queue.mark_running = lambda id: self.queue.cancel(id) and original(id)
        self.assertTrue(self.worker.run_one())
        self.assertEqual(self.queue.get(job["id"])["status"], "cancelled")
        self.assertEqual(self.events(), [])

    def test_without_swapping_the_everyday_model_is_left_alone(self):
        self.settings.update(big_swap_models=False, big_idle_stop_minutes=0)
        self.job()
        self.worker.run_one()
        self.assertEqual(self.small.stops, 0)
        self.assertEqual(self.small.starts, 0)
        self.assertEqual(self.small.state, "ready")
        self.assertEqual(self.events(), [("big", "start"), ("big", "complete"), ("big", "stop")])

    def test_the_big_server_stays_up_for_the_idle_minutes_when_not_swapping(self):
        self.settings.update(big_swap_models=False, big_idle_stop_minutes=3)
        self.job("first")
        self.job("second")
        self.worker.run_one()
        self.assertEqual(self.big.state, "ready")  # kept warm
        self.worker._tick()
        self.assertEqual(self.big.state, "ready")
        self.world.now += datetime.timedelta(minutes=2, seconds=59)
        self.worker._tick()
        self.assertEqual(self.big.state, "ready")
        self.world.now += datetime.timedelta(seconds=2)
        self.worker._tick()
        self.assertEqual(self.big.state, "stopped")
        self.assertEqual(self.big.starts, 1)
        self.assertEqual(self.queue.counts()["queued"], 1)

    def test_a_warm_big_server_is_reused_for_the_next_job(self):
        self.settings.update(big_swap_models=False)
        self.job("first")
        self.job("second")
        self.worker.run_one()
        self.worker.run_one()
        self.assertEqual(self.big.starts, 1)
        self.assertEqual(self.queue.counts()["done"], 2)

    def test_the_idle_timer_restarts_after_each_job(self):
        self.settings.update(big_swap_models=False, big_idle_stop_minutes=3)
        self.job("first")
        self.worker.run_one()
        self.world.now += datetime.timedelta(minutes=2)
        self.job("second")
        self.worker.run_one()
        self.world.now += datetime.timedelta(minutes=2)
        self.worker._tick()
        self.assertEqual(self.big.state, "ready")
        self.world.now += datetime.timedelta(minutes=1)
        self.worker._tick()
        self.assertEqual(self.big.state, "stopped")

    def test_a_model_that_never_finishes_loading_fails_the_job(self):
        self.big.load_polls = 10**9
        self.job()
        self.worker.run_one()
        job = self.queue.list()[0]
        self.assertEqual(job["status"], "error")
        self.assertIn("did not finish loading within 45 minutes", job["error"])
        self.assertEqual(len(self.sleeps), escalation.LOAD_TIMEOUT_SECONDS // escalation.LOAD_POLL_SECONDS)
        self.assertEqual(self.big.state, "stopped")
        self.assertEqual(self.small.starts, 1)


class FailureTests(WorkerCase):
    def assertRestored(self):
        self.assertEqual(self.big.state, "stopped")
        self.assertIn(self.small.state, ("loading", "ready"))
        self.assertFalse(self.worker.status()["small_was_stopped"])
        self.assertEqual(self.queue.counts()["running"], 0)

    def test_model_error_fails_the_job_and_restores_the_servers(self):
        self.world.error = llm.LLMError("cannot reach llama-server: boom")
        job = self.job(exchange_id=5)
        self.assertTrue(self.worker.run_one())
        failed = self.queue.get(job["id"])
        self.assertEqual((failed["status"], failed["answer"]), ("error", ""))
        self.assertIn("boom", failed["error"])
        self.assertEqual(self.exchanges.teachers, {})
        self.assertIn("boom", self.worker.status()["last_error"])
        self.assertRestored()

    def test_unexpected_exceptions_also_fail_the_job(self):
        self.world.error = KeyError("surprise")
        job = self.job()
        self.worker.run_one()
        self.assertEqual(self.queue.get(job["id"])["status"], "error")
        self.assertRestored()

    def test_a_server_that_cannot_start_fails_the_job(self):
        self.big.start_error = "llama-server is not set up yet."
        job = self.job()
        self.worker.run_one()
        self.assertEqual(self.queue.get(job["id"])["error"], "llama-server is not set up yet.")
        self.assertEqual(self.world.calls, [])
        self.assertRestored()

    def test_a_server_dying_while_loading_fails_the_job(self):
        self.big.load_polls = 2
        self.big.die_while_loading = "llama-server exited with code 1. See the log below."
        job = self.job()
        self.worker.run_one()
        self.assertIn("exited with code 1", self.queue.get(job["id"])["error"])
        self.assertEqual(self.world.calls, [])
        self.assertRestored()

    def test_a_server_stopping_by_itself_while_loading_fails_the_job(self):
        self.big.load_polls = 5
        original = self.big.status

        def vanish():
            if self.big.state == "loading":
                self.big.state = "stopped"
            return original()

        self.big.status = vanish
        job = self.job()
        self.worker.run_one()
        self.assertIn("stopped before it finished loading", self.queue.get(job["id"])["error"])
        self.assertRestored()

    def test_an_empty_answer_is_an_error(self):
        self.world.reply["content"] = "  \n "
        job = self.job(exchange_id=2)
        self.worker.run_one()
        failed = self.queue.get(job["id"])
        self.assertEqual(failed["status"], "error")
        self.assertIn("empty answer", failed["error"])
        self.assertEqual(self.exchanges.teachers, {})
        self.assertRestored()

    def test_no_big_model_selected(self):
        self.settings["big_model_path"] = ""
        job = self.job()
        self.worker.run_one()
        self.assertIn("No big model", self.queue.get(job["id"])["error"])
        self.assertEqual(self.events(), [])  # nothing was stopped or started
        self.assertEqual(self.small.state, "ready")

    def test_a_failing_teacher_write_rolls_the_job_back_to_an_error(self):
        self.exchanges.fail_with = RuntimeError("disk full")
        job = self.job(exchange_id=4)
        self.worker.run_one()
        failed = self.queue.get(job["id"])
        self.assertEqual((failed["status"], failed["answer"], failed["error"]), ("error", "", "disk full"))
        self.assertRestored()

    def test_the_big_port_must_differ_from_the_everyday_port(self):
        self.settings.update(big_port=8080, big_swap_models=False)
        job = self.job()
        self.worker.run_one()
        self.assertIn("is the everyday model's port", self.queue.get(job["id"])["error"])
        self.assertEqual(self.big.starts, 0)
        self.assertEqual(self.world.calls, [])

    def test_the_same_port_is_fine_when_the_everyday_model_is_stopped_first(self):
        self.settings.update(big_port=8080, big_swap_models=True)
        job = self.job()
        self.worker.run_one()
        self.assertEqual(self.queue.get(job["id"])["status"], "done")
        self.assertEqual(self.big.settings[0]["port"], 8080)

    def test_user_restarting_the_everyday_model_meanwhile_is_not_an_error(self):
        job = self.job()
        self.world.inside = lambda: setattr(self.small, "state", "ready")
        self.worker.run_one()
        self.assertEqual(self.queue.get(job["id"])["status"], "done")
        self.assertEqual(self.small.starts, 0)
        self.assertEqual(self.worker.status()["last_error"], "")

    def test_failing_to_restart_the_everyday_model_is_reported_not_raised(self):
        job = self.job()
        self.small.start_error = "No model selected."
        self.assertTrue(self.worker.run_one())
        self.assertEqual(self.queue.get(job["id"])["status"], "done")
        self.assertIn("Could not restart the everyday model: No model selected.", self.worker.status()["last_error"])

    def test_a_later_success_clears_the_last_error(self):
        self.world.error = llm.LLMError("flaky")
        self.job("a")
        self.worker.run_one()
        self.assertIn("flaky", self.worker.status()["last_error"])
        self.world.error = None
        self.job("b")
        self.worker.run_one()
        self.assertEqual(self.worker.status()["last_error"], "")

    def test_one_bad_job_does_not_stop_the_next(self):
        self.world.error = llm.LLMError("nope")
        bad = self.job("a")
        self.worker.run_one()
        self.world.error = None
        good = self.job("b")
        self.worker.run_one()
        self.assertEqual((self.queue.get(bad["id"])["status"], self.queue.get(good["id"])["status"]),
                         ("error", "done"))


class ShouldRunTests(WorkerCase):
    def test_nothing_queued_never_runs(self):
        self.worker.run_now()
        for mode in ("manual", "immediate", "overnight"):
            self.settings["big_run_mode"] = mode
            self.assertFalse(self.worker.should_run(), mode)

    def test_matrix(self):
        self.job()
        inside, outside = datetime.datetime(2025, 1, 6, 3, 0), datetime.datetime(2025, 1, 6, 12, 0)
        cases = [("manual", outside, False), ("manual", inside, False), ("immediate", outside, True),
                 ("immediate", inside, True), ("overnight", inside, True), ("overnight", outside, False),
                 ("bogus", inside, False)]
        for mode, now, expected in cases:
            self.settings["big_run_mode"], self.world.now = mode, now
            self.assertEqual(self.worker.should_run(), expected, (mode, now))

    def test_run_now_overrides_every_mode(self):
        self.job()
        self.worker.run_now()
        for mode in ("manual", "immediate", "overnight"):
            self.settings["big_run_mode"] = mode
            self.assertTrue(self.worker.should_run(), mode)

    def test_overnight_with_a_bad_window_never_runs(self):
        self.job()
        self.settings.update(big_run_mode="overnight", overnight_start="late", overnight_end="")
        self.assertFalse(self.worker.should_run())

    def test_only_waiting_jobs_count(self):
        job = self.job()
        self.settings["big_run_mode"] = "immediate"
        self.queue.mark_running(job["id"])
        self.assertFalse(self.worker.should_run())
        self.queue.finish(job["id"], "x")
        self.assertFalse(self.worker.should_run())

    def test_window_wrapping_midnight(self):
        self.job()
        self.settings.update(big_run_mode="overnight", overnight_start="22:00", overnight_end="06:00")
        for hour, expected in ((23, True), (2, True), (6, False), (12, False)):
            self.world.now = datetime.datetime(2025, 1, 6, hour, 0)
            self.assertEqual(self.worker.should_run(), expected, hour)


class TickTests(WorkerCase):
    def test_run_now_drains_the_whole_queue_with_one_model_load(self):
        for i in range(3):
            self.job(f"question {i}")
        self.worker.run_now()
        self.assertTrue(self.worker.status()["run_now"])
        self.worker._tick()
        self.assertEqual(self.queue.counts()["done"], 3)
        self.assertEqual((self.small.stops, self.big.starts, self.small.starts), (1, 1, 1))
        self.assertEqual(len(self.world.calls), 3)
        self.assertFalse(self.worker.status()["run_now"])
        self.job("a new one")  # run_now was one-shot: manual mode waits again
        self.assertFalse(self.worker.should_run())
        self.assertEqual(self.worker.status()["phase"], "waiting")

    def test_manual_mode_does_nothing_without_run_now(self):
        self.job()
        self.worker._tick()
        self.assertEqual(self.queue.counts()["queued"], 1)
        self.assertEqual(self.events(), [])

    def test_jobs_run_by_priority_then_age(self):
        review = self.job("review", reason="review", priority=-1)
        old = self.job("old")
        urgent = self.job("urgent", priority=5)
        self.settings["big_run_mode"] = "immediate"
        self.worker._tick()
        asked = [c["messages"][-1]["content"] for c in self.world.calls]
        self.assertEqual(asked, ["urgent", "old", "review"])
        self.assertEqual({self.queue.get(j["id"])["status"] for j in (review, old, urgent)}, {"done"})

    def test_overnight_jobs_stop_at_the_end_of_the_window(self):
        self.settings.update(big_run_mode="overnight")
        self.world.now = datetime.datetime(2025, 1, 6, 6, 59)
        first, second = self.job("first"), self.job("second")

        def window_closes():
            self.world.now = datetime.datetime(2025, 1, 6, 7, 1)

        self.world.inside = window_closes
        self.worker._tick()
        self.assertEqual(self.queue.get(first["id"])["status"], "done")
        self.assertEqual(self.queue.get(second["id"])["status"], "queued")
        self.assertEqual(self.worker.status()["window_active"], False)
        self.assertEqual(self.small.state, "loading")  # the everyday model is back for the day

    def test_status_reports_waiting_and_window(self):
        self.assertEqual(self.worker.status()["phase"], "idle")
        self.job()
        status = self.worker.status()
        self.assertEqual((status["phase"], status["window_active"], status["big_state"], status["run_now"]),
                         ("waiting", False, "stopped", False))
        self.assertEqual(status["counts"]["queued"], 1)
        self.world.now = datetime.datetime(2025, 1, 6, 2, 0)
        self.assertTrue(self.worker.status()["window_active"])
        self.assertEqual(set(status), {"phase", "current_job", "run_now", "window_active", "big_state", "counts",
                                       "small_was_stopped", "last_error"})
        json.dumps(status)

    def test_status_is_idle_when_a_job_is_due(self):
        self.settings["big_run_mode"] = "immediate"
        self.job()
        self.assertEqual(self.worker.status()["phase"], "idle")


class ReviewTests(WorkerCase):
    def exchange(self, i, question="Q"):
        return {"id": i, "chat_id": f"chat{i}", "messages": convo(f"{question} {i}")}

    def test_schedule_review_queues_recent_answers_at_low_priority(self):
        self.exchanges.recent = [self.exchange(i) for i in (3, 2, 1)]
        self.assertEqual(self.worker.schedule_review(2), 2)
        jobs = self.queue.list()
        self.assertEqual({(j["exchange_id"], j["reason"], j["priority"], j["chat_id"]) for j in jobs},
                         {(3, "review", -1, "chat3"), (2, "review", -1, "chat2")})

    def test_schedule_review_does_not_requeue_what_is_already_waiting(self):
        self.exchanges.recent = [self.exchange(1), self.exchange(2)]
        self.assertEqual(self.worker.schedule_review(5), 2)
        self.assertEqual(self.worker.schedule_review(5), 0)
        self.assertEqual(self.queue.counts()["queued"], 2)

    def test_schedule_review_respects_the_queue_limit(self):
        self.settings["queue_max"] = 2
        self.exchanges.recent = [self.exchange(i) for i in range(5)]
        self.assertEqual(self.worker.schedule_review(5), 2)

    def test_schedule_review_skips_exchanges_without_a_question(self):
        self.exchanges.recent = [{"id": 1, "chat_id": "c", "messages": []}, self.exchange(2)]
        self.assertEqual(self.worker.schedule_review(5), 1)

    def test_schedule_review_with_nothing_to_do(self):
        self.assertEqual(self.worker.schedule_review(5), 0)
        self.assertEqual(self.worker.schedule_review(0), 0)
        self.assertEqual(self.worker.schedule_review(-3), 0)

    def test_reviews_run_after_everything_else_and_become_teacher_answers(self):
        self.exchanges.recent = [self.exchange(8)]
        self.worker.schedule_review(1)
        manual = self.job("manual")
        self.settings["big_run_mode"] = "immediate"
        self.worker._tick()
        self.assertEqual([c["messages"][-1]["content"] for c in self.world.calls], ["manual", "Q 8"])
        self.assertIn(8, self.exchanges.teachers)
        self.assertEqual(self.queue.get(manual["id"])["status"], "done")


class NightlyReviewTests(WorkerCase):
    def setUp(self):
        super().setUp()
        self.settings.update(overnight_review_count=2)
        self.exchanges.recent = [{"id": i, "chat_id": f"c{i}", "messages": convo(f"Q{i}")} for i in (1, 2, 3)]

    def at(self, day, hour, minute=0):
        self.world.now = datetime.datetime(2025, 1, day, hour, minute)

    def test_only_inside_the_window(self):
        self.at(6, 12)
        self.assertEqual(self.worker.maybe_nightly_review(), 0)
        self.at(6, 0, 59)
        self.assertEqual(self.worker.maybe_nightly_review(), 0)
        self.at(6, 7)
        self.assertEqual(self.worker.maybe_nightly_review(), 0)
        self.assertEqual(self.queue.counts()["queued"], 0)

    def test_once_per_night(self):
        self.at(6, 2)
        self.assertEqual(self.worker.maybe_nightly_review(), 2)
        self.at(6, 3)
        self.assertEqual(self.worker.maybe_nightly_review(), 0)
        self.at(6, 6, 59)
        self.assertEqual(self.worker.maybe_nightly_review(), 0)
        self.assertEqual(self.queue.counts()["queued"], 2)

    def test_again_on_the_next_night(self):
        self.at(6, 2)
        self.worker.maybe_nightly_review()
        for job in self.queue.list():
            self.queue.cancel(job["id"])
        self.at(7, 2)
        self.assertEqual(self.worker.maybe_nightly_review(), 2)

    def test_a_window_wrapping_midnight_is_one_night(self):
        self.settings.update(overnight_start="22:00", overnight_end="06:00")
        self.at(6, 23)
        self.assertEqual(self.worker.maybe_nightly_review(), 2)
        for job in self.queue.list():
            self.queue.cancel(job["id"])
        self.at(7, 1)  # after midnight, still the night that began on the 6th
        self.assertEqual(self.worker.maybe_nightly_review(), 0)
        self.at(7, 23)
        self.assertEqual(self.worker.maybe_nightly_review(), 2)

    def test_disabled_without_a_count_or_a_big_model(self):
        self.at(6, 2)
        self.settings["overnight_review_count"] = 0
        self.assertEqual(self.worker.maybe_nightly_review(), 0)
        self.settings.update(overnight_review_count=2, big_model_path="")
        self.assertEqual(self.worker.maybe_nightly_review(), 0)
        self.settings["big_model_path"] = BIG_PATH
        self.assertEqual(self.worker.maybe_nightly_review(), 2)  # the skipped checks did not use up the night

    def test_an_empty_night_is_not_retried(self):
        self.exchanges.recent = []
        self.at(6, 2)
        self.assertEqual(self.worker.maybe_nightly_review(), 0)
        self.exchanges.recent = [{"id": 9, "chat_id": "c", "messages": convo("late arrival")}]
        self.at(6, 4)
        self.assertEqual(self.worker.maybe_nightly_review(), 0)

    def test_the_loop_does_the_nightly_review_and_runs_it_in_overnight_mode(self):
        self.settings["big_run_mode"] = "overnight"
        self.at(6, 2)
        self.worker._tick()
        self.assertEqual(self.queue.counts()["done"], 2)
        self.assertEqual(sorted(self.exchanges.teachers), [1, 2])


class ThreadTests(WorkerCase):
    def wait_for(self, condition, seconds=10):
        deadline = time.time() + seconds
        while time.time() < deadline:
            if condition():
                return True
            time.sleep(0.02)
        return False

    def test_start_recovers_jobs_stuck_in_running(self):
        stuck = self.job()
        self.queue.mark_running(stuck["id"])
        self.worker.start()
        self.worker.stop()
        self.assertEqual(self.queue.get(stuck["id"])["status"], "queued")

    def test_the_thread_processes_the_queue_when_woken(self):
        self.settings["big_run_mode"] = "immediate"
        self.worker.start()
        job = self.job(exchange_id=1)
        self.worker.wake()
        self.assertTrue(self.wait_for(lambda: self.queue.get(job["id"])["status"] == "done"))
        self.assertTrue(self.wait_for(lambda: self.small.state != "stopped"
                                      and self.worker.status()["phase"] == "idle"))
        self.worker.stop()
        self.assertEqual(self.exchanges.teachers[1][0], "The big answer.")

    def test_run_now_wakes_the_thread(self):
        self.worker.start()
        job = self.job()
        self.worker.run_now()
        self.assertTrue(self.wait_for(lambda: self.queue.get(job["id"])["status"] == "done"))
        self.worker.stop()

    def test_start_twice_runs_one_thread_and_stop_is_idempotent(self):
        self.worker.start()
        thread = self.worker._thread
        self.worker.start()
        self.assertIs(self.worker._thread, thread)
        self.worker.stop()
        self.worker.stop()
        self.assertFalse(thread.is_alive())

    def test_the_worker_can_be_restarted(self):
        self.worker.start()
        self.worker.stop()
        self.settings["big_run_mode"] = "immediate"
        job = self.job()
        self.worker.start()
        self.assertTrue(self.wait_for(lambda: self.queue.get(job["id"])["status"] == "done"))
        self.worker.stop()

    def test_stop_shuts_the_big_server_down_and_requeues_the_job_in_flight(self):
        entered, released = threading.Event(), threading.Event()
        self.big.on_stop = released.set

        def block():
            entered.set()
            released.wait(10)
            raise llm.LLMError("cannot reach llama-server: connection reset")

        self.world.inside = block
        self.settings["big_run_mode"] = "immediate"
        job = self.job()
        self.worker.start()
        self.assertTrue(entered.wait(10))
        self.worker.stop()
        self.assertEqual(self.queue.get(job["id"])["status"], "queued")
        self.assertEqual(self.queue.get(job["id"])["started"], 0)
        self.assertEqual(self.big.state, "stopped")
        self.assertFalse(self.worker._thread)
        self.assertEqual(self.small.starts, 0)  # shutting down: the everyday model is not brought back
        self.assertFalse(self.worker.status()["small_was_stopped"])

    def test_stopping_after_the_window_closed_does_not_bring_the_everyday_model_back(self):
        entered, released = threading.Event(), threading.Event()
        self.big.on_stop = released.set

        def window_closes_then_block():
            self.world.now = datetime.datetime(2025, 1, 6, 8, 0)
            entered.set()
            released.wait(10)
            raise llm.LLMError("connection reset")

        self.settings["big_run_mode"] = "overnight"
        self.world.now = datetime.datetime(2025, 1, 6, 3, 0)
        self.world.inside = window_closes_then_block
        self.job()
        self.worker.start()
        self.worker.wake()
        self.assertTrue(entered.wait(10))
        self.worker.stop()
        self.assertEqual((self.small.starts, self.small.state), (0, "stopped"))
        self.assertEqual(self.queue.counts()["queued"], 1)

    def test_the_loop_survives_errors(self):
        calls = []
        original = self.worker.maybe_nightly_review

        def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("database is locked")
            return original()

        self.worker.maybe_nightly_review = flaky
        self.worker.start()
        self.assertTrue(self.wait_for(lambda: "database is locked" in self.worker.status()["last_error"]))
        self.worker.wake()
        self.assertTrue(self.wait_for(lambda: len(calls) >= 2))
        self.assertTrue(self.worker._thread.is_alive())
        self.worker.stop()


class AnswerTimeoutTests(unittest.TestCase):
    def test_timeouts_scale_with_the_answer_length_but_stay_bounded(self):
        self.assertEqual(escalation._answer_timeout(64), 1800)
        self.assertEqual(escalation._answer_timeout(3000), 12000)
        self.assertEqual(escalation._answer_timeout(65536), 6 * 3600)


if __name__ == "__main__":
    unittest.main()
