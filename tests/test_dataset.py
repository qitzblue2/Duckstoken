import ast
import contextlib
import datetime
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["DUCKSTOKEN_HOME"] = tempfile.mkdtemp(prefix="duck-dataset-")

from duckstoken import __version__, dataset  # noqa: E402
from duckstoken.dataset import DatasetStore, redact, train_script, training_readme  # noqa: E402
from duckstoken.db import Database  # noqa: E402
from duckstoken.feedback import ExchangeStore  # noqa: E402

LONG_ANSWER = "Boil the egg for ten minutes, then cool it in cold water."


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Database(Path(self.tmp.name) / "x.db")
        self.addCleanup(self.db.close)  # Windows cannot delete a database that is still open
        self.exchanges = ExchangeStore(self.db)
        self.store = DatasetStore(self.db, self.exchanges)
        self.n = 0

    def exchange(self, prompt="How do I boil an egg?", answer=LONG_ANSWER, *, rating=0, correction="", teacher="",
                 messages=None, reasoning=""):
        """Insert an exchange row directly and return its id."""
        self.n += 1
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        cur = self.db.execute(
            """INSERT INTO exchanges(chat_id, msg_index, created, prompt, messages_json, answer, reasoning, rating,
                                     correction, teacher_answer) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("chat", self.n, time.time(), prompt, json.dumps(messages), answer, reasoning, rating, correction, teacher))
        return cur.lastrowid

    def items(self, **kw):
        return self.store.list(limit=200, **kw)["items"]

    def only_item(self, kind):
        found = [i for i in self.items() if i["kind"] == kind]
        self.assertEqual(len(found), 1, found)
        return found[0]


# ------------------------------------------------------------------------------------------------ refresh
class RefreshTests(StoreCase):
    def test_liked_answer_becomes_an_sft_item(self):
        ex = self.exchange(rating=1)
        self.assertEqual(self.store.refresh(), {"added": 1, "total": 1})
        item = self.only_item("sft")
        self.assertEqual(item["exchange_id"], ex)
        self.assertEqual(item["source"], "liked")
        self.assertEqual(item["status"], "pending")
        self.assertEqual(item["chosen"], LONG_ANSWER)
        self.assertEqual(item["rejected"], "")
        self.assertFalse(item["edited"])
        self.assertEqual(item["messages"], [{"role": "user", "content": "How do I boil an egg?"}])
        self.assertNotIn("messages_json", item)
        self.assertEqual([i["kind"] for i in self.items()], ["sft"])  # a liked answer has no worse sibling

    def test_exchanges_without_a_better_answer_yield_nothing(self):
        self.exchange(rating=0)
        self.exchange(rating=-1)
        self.exchange(rating=1, answer="")
        self.assertEqual(self.store.refresh(), {"added": 0, "total": 0})

    def test_correction_beats_teacher_beats_liked(self):
        self.exchange(rating=1, teacher="The teacher's answer.", correction="The user's own correction.")
        self.exchange(rating=1, teacher="The teacher's answer.")
        self.exchange(rating=1)
        self.store.refresh()
        by_source = {i["source"]: i["chosen"] for i in self.items(kind="sft")}
        self.assertEqual(by_source, {"corrected": "The user's own correction.", "teacher": "The teacher's answer.",
                                     "liked": LONG_ANSWER})

    def test_preference_pair_uses_the_original_answer_as_rejected(self):
        self.exchange(rating=-1, correction="Boil it for 10 minutes and chill it.")
        self.exchange(rating=0, teacher="A better answer from the big model.")
        self.assertEqual(self.store.refresh(), {"added": 4, "total": 4})
        prefs = self.items(kind="pref")
        self.assertEqual(sorted(p["source"] for p in prefs), ["corrected", "teacher"])
        for pref in prefs:
            self.assertEqual(pref["rejected"], LONG_ANSWER)
            self.assertNotEqual(pref["chosen"], pref["rejected"])
            self.assertEqual(pref["messages"], [{"role": "user", "content": "How do I boil an egg?"}])
        self.assertEqual(len(self.items(kind="sft")), 2)

    def test_no_preference_pair_when_the_better_answer_is_the_same_text(self):
        self.exchange(correction="  boil the EGG for ten minutes,\nthen cool it in cold water. ")
        self.exchange(teacher=LONG_ANSWER, answer="")
        self.store.refresh()
        self.assertEqual([i["kind"] for i in self.items()], ["sft", "sft"])

    def test_short_chosen_answers_are_skipped(self):
        self.exchange(rating=1, answer="Yes, ok.")       # 8 characters: kept
        self.exchange(rating=1, answer="Yes ok.")        # 7: skipped
        self.exchange(correction="  Short.  ")           # 6 after trimming: skipped
        self.store.refresh()
        self.assertEqual([i["chosen"] for i in self.items()], ["Yes, ok."])

    def test_refresh_is_idempotent(self):
        self.exchange(rating=1)
        self.exchange(correction="Boil it for 10 minutes and chill it.")
        first = self.store.refresh()
        self.assertEqual(first, {"added": 3, "total": 3})
        before = self.items()
        self.assertEqual(self.store.refresh(), {"added": 0, "total": 3})
        self.assertEqual(self.items(), before)

    def test_more_exchanges_than_one_scan_page(self):
        for i in range(dataset.SCAN_PAGE * 2 + 7):
            self.exchange(f"Question {i}?", rating=1, answer=f"Answer number {i}.")
        self.assertEqual(self.store.refresh(), {"added": dataset.SCAN_PAGE * 2 + 7, "total": dataset.SCAN_PAGE * 2 + 7})
        self.assertEqual(self.store.refresh()["added"], 0)

    def test_conversation_context_is_cleaned(self):
        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": [{"type": "text", "text": "What is the weather?"}]},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "sunny", "tool_call_id": "1"},
            {"role": "assistant", "content": "It is sunny.", "reasoning": "secret thoughts"},
            {"role": "user", "content": "  And tomorrow?  "},
        ]
        self.exchange(messages=messages, rating=1)
        self.store.refresh()
        self.assertEqual(self.only_item("sft")["messages"], [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "What is the weather?"},
            {"role": "assistant", "content": "It is sunny."},
            {"role": "user", "content": "And tomorrow?"},
        ])

    def test_answers_that_followed_tool_output_are_skipped(self):
        messages = [{"role": "user", "content": "What is the weather?"},
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
                    {"role": "tool", "content": "sunny", "tool_call_id": "1"}]
        self.exchange(messages=messages, rating=1)
        self.exchange(messages=[], rating=1)
        self.exchange(messages=[{"role": "user", "content": "   "}], rating=1)
        broken = self.exchange(rating=1)
        self.db.execute("UPDATE exchanges SET messages_json = 'not json' WHERE id = ?", (broken,))
        self.assertEqual(self.store.refresh(), {"added": 0, "total": 0})

    def test_pending_items_follow_their_exchange(self):
        ex = self.exchange(rating=1)
        self.store.refresh()
        self.db.execute("UPDATE exchanges SET correction = ? WHERE id = ?", ("A much better answer, corrected.", ex))
        self.assertEqual(self.store.refresh(), {"added": 1, "total": 2})  # the new pref item; the sft one is updated
        sft = self.only_item("sft")
        self.assertEqual((sft["chosen"], sft["source"]), ("A much better answer, corrected.", "corrected"))
        self.assertEqual(self.only_item("pref")["rejected"], LONG_ANSWER)

    def test_approved_rejected_and_edited_items_are_never_overwritten(self):
        a = self.exchange("Approved question?", rating=1, answer="The approved answer.")
        r = self.exchange("Rejected question?", rating=1, answer="The rejected answer.")
        e = self.exchange("Edited question?", rating=1, answer="The edited answer.")
        self.store.refresh()
        ids = {i["exchange_id"]: i["id"] for i in self.items()}
        self.store.set_status([ids[a]], "approved")
        self.store.set_status([ids[r]], "rejected")
        self.store.edit(ids[e], "My own wording of the answer.")
        before = {i["id"]: i for i in self.items()}
        for ex in (a, r, e):
            self.db.execute("UPDATE exchanges SET correction = 'Something else entirely.', answer = 'New.' "
                            "WHERE id = ?", (ex,))
        self.store.refresh()
        after = {i["id"]: i for i in self.items() if i["kind"] == "sft"}
        self.assertEqual({k: after[k] for k in before}, before)
        # the new corrections added preference items next to them, which is not "overwriting"
        self.assertEqual(len([i for i in self.items() if i["kind"] == "pref"]), 3)

    def test_rejected_items_are_not_added_again(self):
        self.exchange(rating=1)
        self.store.refresh()
        item = self.only_item("sft")
        self.store.set_status([item["id"]], "rejected")
        self.assertEqual(self.store.refresh(), {"added": 0, "total": 1})
        self.assertEqual(self.only_item("sft")["status"], "rejected")

    def test_untouched_pending_items_vanish_with_their_feedback(self):
        gone = self.exchange("Gone question?", rating=1)
        unrated = self.exchange("Unrated question?", rating=1)
        kept = self.exchange("Kept question?", rating=1)
        approved = self.exchange("Approved question?", rating=1)
        self.store.refresh()
        approved_item = next(i for i in self.items() if i["exchange_id"] == approved)
        self.store.set_status([approved_item["id"]], "approved")
        self.db.execute("DELETE FROM exchanges WHERE id IN (?, ?)", (gone, approved))
        self.db.execute("UPDATE exchanges SET rating = 0 WHERE id = ?", (unrated,))
        self.assertEqual(self.store.refresh(), {"added": 0, "total": 2})
        self.assertEqual(sorted(i["exchange_id"] for i in self.items()), sorted([kept, approved]))

    def test_deleted_items_come_back_while_the_exchange_qualifies(self):
        self.exchange(rating=1)
        self.store.refresh()
        self.assertTrue(self.store.delete(self.only_item("sft")["id"]))
        self.assertEqual(self.store.refresh(), {"added": 1, "total": 1})

    def test_concurrent_refreshes_do_not_duplicate_or_fail(self):
        for i in range(30):
            self.exchange(f"Question {i}?", rating=1, answer=f"Answer number {i}.")
        errors = []

        def work():
            try:
                self.store.refresh()
            except Exception as e:  # pragma: no cover - reported below
                errors.append(e)

        threads = [threading.Thread(target=work) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(self.store.stats()["total"], 30)


# ------------------------------------------------------------------------------------------------ reviewing
class ReviewTests(StoreCase):
    def setUp(self):
        super().setUp()
        self.exchange("First question?", rating=1, answer="First answer, liked.")
        self.exchange("Second question?", correction="Second answer, corrected.")
        self.exchange("Third_question 100%?", rating=1, answer="Third answer with 100% and a_b.")
        self.store.refresh()  # 3 sft + 1 pref

    def test_list_filters_paginates_and_orders_newest_first(self):
        page = self.store.list()
        self.assertEqual(page["total"], 4)
        ids = [i["id"] for i in page["items"]]
        self.assertEqual(ids, sorted(ids, reverse=True))
        self.assertEqual(self.store.list(kind="pref")["total"], 1)
        self.assertEqual(self.store.list(kind="sft")["total"], 3)
        self.assertEqual(self.store.list(status="approved"), {"items": [], "total": 0})
        self.assertEqual(self.store.list(status="nonsense"), {"items": [], "total": 0})
        first = self.store.list(limit=3, offset=0)
        second = self.store.list(limit=3, offset=3)
        self.assertEqual((len(first["items"]), len(second["items"])), (3, 1))
        self.assertEqual(sorted(i["id"] for i in first["items"] + second["items"]), sorted(ids))
        self.assertEqual(len(self.store.list(limit=-5)["items"]), 1)
        self.assertEqual(self.store.list(limit="x", offset="y")["total"], 4)

    def test_search_covers_answers_and_prompts_and_escapes_like_wildcards(self):
        self.assertEqual(self.store.list(q="second question")["total"], 2)   # sft + pref share the prompt
        self.assertEqual(self.store.list(q="LIKED")["total"], 1)
        self.assertEqual(self.store.list(q="100%")["total"], 1)
        self.assertEqual(self.store.list(q="%")["total"], 1)                 # a literal percent sign, not a wildcard
        self.assertEqual(self.store.list(q="a_b")["total"], 1)
        self.assertEqual(self.store.list(q="_")["total"], 1)                 # a literal underscore, not "any char"
        self.assertEqual(self.store.list(q="nothing like this")["total"], 0)
        for hostile in ('"', "'", "*", "OR", "NEAR(", ")", "-", "\\", "\x00", "💥",
                        "'; DROP TABLE dataset_items; --"):
            self.store.list(q=hostile)
        self.assertEqual(self.store.stats()["total"], 4)

    def test_get(self):
        item = self.items()[0]
        self.assertEqual(self.store.get(item["id"]), item)
        for bad in (0, -1, 10**9, 2**40, None, "1", 1.0, True, [1]):
            self.assertIsNone(self.store.get(bad), bad)

    def test_set_status(self):
        ids = [i["id"] for i in self.items()]
        self.assertEqual(self.store.set_status(ids[:2], "approved"), 2)
        self.assertEqual(self.store.set_status([ids[0], ids[0], 999, "x", None, True, -3], "rejected"), 1)
        self.assertEqual(self.store.get(ids[0])["status"], "rejected")
        self.assertEqual(self.store.get(ids[1])["status"], "approved")
        self.assertEqual(self.store.set_status([], "pending"), 0)
        self.assertEqual(self.store.set_status(tuple(ids), "pending"), 4)

    def test_set_status_validation(self):
        with self.assertRaises(ValueError):
            self.store.set_status([1], "done")
        with self.assertRaises(ValueError):
            self.store.set_status([1], None)
        for bad in (None, 5, "12", {1: 2}):
            with self.assertRaises(ValueError, msg=bad):
                self.store.set_status(bad, "approved")
        with self.assertRaises(ValueError):
            self.store.set_status(list(range(dataset.MAX_IDS + 1)), "approved")
        self.assertEqual(self.store.stats()["pending"], 4)

    def test_set_status_handles_more_ids_than_one_sql_chunk(self):
        for i in range(dataset.SQL_CHUNK + 50):
            self.db.execute("INSERT INTO dataset_items(exchange_id, kind, messages_json, chosen, source, created) "
                            "VALUES (?, 'sft', '[]', 'x', 'liked', 0)", (1000 + i,))
        every = [r["id"] for r in self.db.query("SELECT id FROM dataset_items")]
        self.assertGreater(len(every), dataset.SQL_CHUNK)
        self.assertEqual(self.store.set_status(every, "approved"), len(every))
        self.assertEqual(self.store.stats()["approved"], len(every))

    def test_edit_sft_item(self):
        item = next(i for i in self.items() if i["kind"] == "sft")
        edited = self.store.edit(item["id"], "  A better wording of the answer.  ", rejected="ignored for sft items")
        self.assertEqual(edited["chosen"], "A better wording of the answer.")
        self.assertEqual(edited["rejected"], "")
        self.assertTrue(edited["edited"])
        self.assertEqual(edited["status"], "pending")
        self.assertEqual(edited["messages"], item["messages"])

    def test_edit_pref_item(self):
        item = next(i for i in self.items() if i["kind"] == "pref")
        kept = self.store.edit(item["id"], "New chosen answer.")
        self.assertEqual((kept["chosen"], kept["rejected"]), ("New chosen answer.", item["rejected"]))
        both = self.store.edit(item["id"], "Newer chosen answer.", rejected="A clearly worse answer.")
        self.assertEqual((both["chosen"], both["rejected"]), ("Newer chosen answer.", "A clearly worse answer."))

    def test_edit_validation(self):
        sft = next(i for i in self.items() if i["kind"] == "sft")
        pref = next(i for i in self.items() if i["kind"] == "pref")
        for bad in ("", "   \n", None, 5, ["x"]):
            with self.assertRaises(ValueError, msg=bad):
                self.store.edit(sft["id"], bad)
        with self.assertRaises(ValueError):
            self.store.edit(sft["id"], "x" * (dataset.MAX_TEXT + 1))
        with self.assertRaises(ValueError):
            self.store.edit(pref["id"], "Some answer.", rejected="")
        with self.assertRaises(ValueError):
            self.store.edit(pref["id"], "Some answer.", rejected=7)
        with self.assertRaises(ValueError):
            self.store.edit(pref["id"], "same  ANSWER", rejected="Same answer")
        with self.assertRaises(ValueError):  # identical to the stored rejected answer
            self.store.edit(pref["id"], pref["rejected"])
        self.assertEqual(self.store.get(sft["id"])["chosen"], sft["chosen"])
        self.assertFalse(self.store.get(pref["id"])["edited"])
        self.assertIsNone(self.store.edit(99999, "Some answer."))
        self.assertIsNone(self.store.edit("x", "Some answer."))

    def test_edit_cleans_unsafe_characters(self):
        item = self.items()[0]
        edited = self.store.edit(item["id"], "bad\x00 text \ud800 here")
        self.assertEqual(edited["chosen"], "bad text ? here")
        self.assertEqual(self.store.edit(item["id"], "x" * dataset.MAX_TEXT)["chosen"], "x" * dataset.MAX_TEXT)

    def test_delete(self):
        item = self.items()[0]
        self.assertTrue(self.store.delete(item["id"]))
        self.assertFalse(self.store.delete(item["id"]))
        self.assertFalse(self.store.delete("x"))
        self.assertIsNone(self.store.get(item["id"]))
        self.assertEqual(self.store.stats()["total"], 3)

    def test_stats(self):
        ids = [i["id"] for i in self.items()]
        self.store.set_status(ids[:1], "approved")
        self.store.set_status(ids[1:2], "rejected")
        self.assertEqual(self.store.stats(),
                         {"pending": 2, "approved": 1, "rejected": 1, "sft": 3, "pref": 1, "total": 4})

    def test_stats_of_an_empty_dataset(self):
        empty = DatasetStore(Database(Path(self.tmp.name) / "empty.db"), self.exchanges)
        self.addCleanup(empty.db.close)
        self.assertEqual(empty.stats(), {"pending": 0, "approved": 0, "rejected": 0, "sft": 0, "pref": 0, "total": 0})
        self.assertEqual(empty.list(), {"items": [], "total": 0})


# ------------------------------------------------------------------------------------------------ export
def unzip(blob: bytes) -> dict[str, str]:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        assert z.testzip() is None
        return {n: z.read(n).decode("utf-8") for n in z.namelist()}


def rows(text: str) -> list[dict]:
    return [json.loads(line) for line in text.split("\n") if line]


class ExportTests(StoreCase):
    def approve_all(self):
        self.store.set_status([i["id"] for i in self.items()], "approved")

    def seed(self):
        self.exchange("What is 2+2?", rating=1, answer="2+2 is 4.")
        self.exchange("Capital of France?", rating=-1, correction="The capital of France is Paris.")
        self.store.refresh()  # 2 sft + 1 pref
        self.approve_all()

    def test_zip_structure_and_content(self):
        self.seed()
        files = unzip(self.store.export_zip())
        self.assertEqual(list(files), ["sft.jsonl", "pref.jsonl", "manifest.json", "README_TRAINING.md",
                                       "train_lora.py"])
        sft, pref = rows(files["sft.jsonl"]), rows(files["pref.jsonl"])
        self.assertEqual(sft, [
            {"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "2+2 is 4."}]},
            {"messages": [{"role": "user", "content": "Capital of France?"},
                          {"role": "assistant", "content": "The capital of France is Paris."}]}])
        self.assertEqual(pref, [{"prompt": [{"role": "user", "content": "Capital of France?"}],
                                 "chosen": [{"role": "assistant", "content": "The capital of France is Paris."}],
                                 "rejected": [{"role": "assistant", "content": LONG_ANSWER}]}])
        self.assertEqual(files["README_TRAINING.md"], training_readme())
        self.assertEqual(files["train_lora.py"], train_script())
        for name in ("sft.jsonl", "pref.jsonl"):
            self.assertTrue(files[name].endswith("\n"))

    def test_manifest(self):
        self.seed()
        self.exchange("Teacher question?", teacher="The big model's answer.")
        self.store.refresh()
        self.approve_all()
        manifest = json.loads(unzip(self.store.export_zip(redact=False))["manifest.json"])
        created = datetime.datetime.strptime(manifest.pop("created"), "%Y-%m-%dT%H:%M:%SZ")
        age = datetime.datetime.now(datetime.timezone.utc) - created.replace(tzinfo=datetime.timezone.utc)
        self.assertLess(abs(age.total_seconds()), 60)
        self.assertEqual(manifest, {
            "duckstoken_version": __version__, "redacted": False, "only_approved": True, "include_reasoning": False,
            "counts": {"sft": 3, "pref": 2},
            "sources": {"sft": {"corrected": 1, "teacher": 1, "liked": 1},
                        "pref": {"corrected": 1, "teacher": 1, "liked": 0}}})
        self.assertTrue(json.loads(unzip(self.store.export_zip())["manifest.json"])["redacted"])

    def test_script_is_optional(self):
        self.seed()
        files = unzip(self.store.export_zip(include_script=False))
        self.assertEqual(list(files), ["sft.jsonl", "pref.jsonl", "manifest.json", "README_TRAINING.md"])

    def test_empty_preference_file_is_still_present(self):
        self.exchange(rating=1)
        self.store.refresh()
        self.approve_all()
        files = unzip(self.store.export_zip())
        self.assertEqual(files["pref.jsonl"], "")
        self.assertEqual(len(rows(files["sft.jsonl"])), 1)

    def test_only_approved_items_by_default(self):
        self.seed()
        self.exchange("Pending question?", rating=1, answer="A pending answer.")
        rejected = self.exchange("Rejected question?", rating=1, answer="A rejected answer.")
        self.store.refresh()
        by_exchange = {i["exchange_id"]: i["id"] for i in self.items()}
        self.store.set_status([by_exchange[rejected]], "rejected")
        text = unzip(self.store.export_zip())["sft.jsonl"]
        self.assertNotIn("pending answer", text)
        self.assertNotIn("rejected answer", text)
        everything = unzip(self.store.export_zip(only_approved=False))
        self.assertIn("A pending answer.", everything["sft.jsonl"])
        self.assertNotIn("A rejected answer.", everything["sft.jsonl"])
        self.assertFalse(json.loads(everything["manifest.json"])["only_approved"])

    def test_nothing_to_export_raises(self):
        with self.assertRaisesRegex(ValueError, "approve"):
            self.store.export_zip()
        with self.assertRaises(ValueError):
            self.store.export_zip(only_approved=False)
        self.exchange(rating=1)
        self.store.refresh()
        with self.assertRaises(ValueError):  # pending only
            self.store.export_zip()
        self.assertEqual(len(rows(unzip(self.store.export_zip(only_approved=False))["sft.jsonl"])), 1)
        self.store.set_status([self.items()[0]["id"]], "rejected")
        with self.assertRaises(ValueError):
            self.store.export_zip(only_approved=False)

    def test_redaction_is_on_by_default_and_covers_everything(self):
        self.exchange("Mail bob@example.com about C:\\Users\\Bob\\notes.txt", rating=-1,
                      correction="I wrote to <bob@example.com> from 192.168.1.20 with token ghp_" + "a1B2c3D4e5" * 4,
                      answer="Original answer for bob@example.com at 10.0.0.1.", reasoning="bob@example.com thinks")
        self.store.refresh()
        self.approve_all()
        redacted = unzip(self.store.export_zip(include_reasoning=True))
        text = redacted["sft.jsonl"] + redacted["pref.jsonl"]
        for leaked in ("bob@example.com", "Bob", "192.168.1.20", "10.0.0.1", "ghp_"):
            self.assertNotIn(leaked, text)
        for placeholder in ("<email>", "<user>", "<ip>", "<secret>"):
            self.assertIn(placeholder, text)
        raw = unzip(self.store.export_zip(redact=False))
        self.assertIn("bob@example.com", raw["sft.jsonl"])
        self.assertIn("192.168.1.20", raw["pref.jsonl"])
        self.assertIn("10.0.0.1", raw["pref.jsonl"])

    def test_reasoning_is_included_only_when_asked_and_only_for_the_models_own_answer(self):
        self.exchange("Liked question?", rating=1, answer="The liked original answer.", reasoning="Liked thoughts.")
        self.exchange("Corrected question?", answer="The wrong original answer.", reasoning="Wrong thoughts.",
                      correction="The user's correction.")
        self.exchange("Edited question?", rating=1, answer="Another original answer.", reasoning="Edited thoughts.")
        self.exchange("No reasoning question?", rating=1, answer="An answer without thoughts.")
        self.store.refresh()
        edited = next(i for i in self.items() if i["chosen"] == "Another original answer.")
        self.store.edit(edited["id"], "Another answer, reworded by the user.")
        self.approve_all()

        plain = unzip(self.store.export_zip())
        self.assertNotIn("thinking", plain["sft.jsonl"] + plain["pref.jsonl"])

        files = unzip(self.store.export_zip(include_reasoning=True))
        answers = {r["messages"][-1]["content"]: r["messages"][-1] for r in rows(files["sft.jsonl"])}
        self.assertEqual(answers["The liked original answer."]["thinking"], "Liked thoughts.")
        self.assertNotIn("thinking", answers["The user's correction."])
        self.assertNotIn("thinking", answers["Another answer, reworded by the user."])
        self.assertNotIn("thinking", answers["An answer without thoughts."])
        pref = rows(files["pref.jsonl"])
        self.assertEqual(len(pref), 1)
        self.assertNotIn("thinking", pref[0]["chosen"][0])
        self.assertEqual(pref[0]["rejected"][0]["thinking"], "Wrong thoughts.")

    def test_output_is_deterministic_and_ordered_by_item_id(self):
        for i in range(5):
            self.exchange(f"Question {i}?", rating=1, answer=f"Answer number {i}.")
        self.store.refresh()
        self.approve_all()
        first = unzip(self.store.export_zip())["sft.jsonl"]
        self.assertEqual(first, unzip(self.store.export_zip())["sft.jsonl"])
        self.assertEqual([r["messages"][0]["content"] for r in rows(first)], [f"Question {i}?" for i in range(5)])

    def test_every_line_is_one_json_object_even_with_unicode_line_separators(self):
        self.exchange("Question\u2028with\u2029odd\x85 separators\nand lines?", rating=1,
                      answer="Answer \u2028 with\r\nbreaks, quotes \" and \\ backslashes, emoji \U0001F986.")
        self.store.refresh()
        self.approve_all()
        text = unzip(self.store.export_zip())["sft.jsonl"]
        self.assertEqual(len(text.splitlines()), 1)
        record = json.loads(text)
        self.assertEqual(record["messages"][0]["content"], "Question\u2028with\u2029odd\x85 separators\nand lines?")
        self.assertIn("\U0001F986", record["messages"][1]["content"])

    def test_unusable_items_are_skipped(self):
        self.seed()
        self.db.execute("INSERT INTO dataset_items(exchange_id, kind, messages_json, chosen, rejected, source, status, "
                        "created) VALUES (9001, 'sft', 'garbage', 'x', '', 'liked', 'approved', 0)")
        self.db.execute("INSERT INTO dataset_items(exchange_id, kind, messages_json, chosen, rejected, source, status, "
                        "created) VALUES (9002, 'pref', '[{\"role\":\"user\",\"content\":\"q\"}]', 'x', '', 'liked', "
                        "'approved', 0)")
        files = unzip(self.store.export_zip())
        self.assertEqual(json.loads(files["manifest.json"])["counts"], {"sft": 2, "pref": 1})

    def test_zip_survives_lone_surrogates(self):
        blob = dataset._zip([("a.txt", "x\ud800y")], datetime.datetime.now(datetime.timezone.utc))
        self.assertEqual(unzip(blob), {"a.txt": "x?y"})
        self.assertEqual(dataset._jsonl([{"a": "x\ud800y"}]).count("\n"), 1)

    def test_zip_entries_have_sane_metadata(self):
        self.seed()
        with zipfile.ZipFile(io.BytesIO(self.store.export_zip())) as z:
            for info in z.infolist():
                self.assertGreaterEqual(info.date_time[0], 2024)
                self.assertEqual(info.compress_type, zipfile.ZIP_DEFLATED)
                self.assertNotIn("/", info.filename)


class WithRealExchangeStoreTests(StoreCase):
    """The same flow the app uses: record, rate, correct and teach through ExchangeStore, then build the dataset."""

    def test_full_flow(self):
        convo = [{"role": "user", "content": [{"type": "text", "text": "How do I boil an egg?"}]}]
        liked = self.exchanges.record("chat-1", 1, convo, "Boil it for ten minutes.", reasoning="Eggs need heat.")
        wrong = self.exchanges.record("chat-1", 3, convo + [{"role": "assistant", "content": "Boil it for ten minutes."},
                                                             {"role": "user", "content": "And an omelette?"}],
                                      "Use a kettle.")
        taught = self.exchanges.record("chat-2", 1, [{"role": "user", "content": "Why is the sky blue?"}], "Paint.")
        tools = self.exchanges.record("chat-3", 3, [{"role": "user", "content": "Weather?"},
                                                     {"role": "assistant", "content": "", "tool_calls": [{"id": "t"}]},
                                                     {"role": "tool", "content": "sunny", "tool_call_id": "t"}],
                                      "It is sunny.", used_tools=True)
        self.exchanges.rate(liked["id"], 1)
        self.exchanges.rate(wrong["id"], -1, "Use a pan and eggs.")
        self.exchanges.correct(wrong["id"], "Whisk eggs and fry them in a pan.")
        self.exchanges.set_teacher(taught["id"], "Rayleigh scattering makes the sky look blue.", "big-model")
        self.exchanges.rate(tools["id"], 1)

        self.assertEqual(self.store.refresh(), {"added": 5, "total": 5})
        self.approve_all()
        files = unzip(self.store.export_zip(include_reasoning=True))
        sft, pref = rows(files["sft.jsonl"]), rows(files["pref.jsonl"])
        self.assertEqual(len(sft), 3)
        self.assertEqual(sft[0]["messages"][0]["content"], "How do I boil an egg?")
        self.assertEqual(sft[0]["messages"][-1]["thinking"], "Eggs need heat.")
        self.assertEqual([r["messages"][-1]["content"] for r in sft[1:]],
                         ["Whisk eggs and fry them in a pan.", "Rayleigh scattering makes the sky look blue."])
        self.assertEqual([p["rejected"][0]["content"] for p in pref], ["Use a kettle.", "Paint."])
        self.assertEqual([m["role"] for m in sft[1]["messages"]], ["user", "assistant", "user", "assistant"])

        self.exchanges.delete_chat("chat-2")
        self.exchanges.delete_chat("chat-1")
        self.assertEqual(self.store.refresh(), {"added": 0, "total": 5})  # approved items outlive their chats

    def approve_all(self):
        self.store.set_status([i["id"] for i in self.items()], "approved")


# ------------------------------------------------------------------------------------------------ redaction
class RedactTests(unittest.TestCase):
    def check(self, table):
        for text, expected in table:
            with self.subTest(text=text):
                self.assertEqual(redact(text), expected)

    def test_emails(self):
        self.check([
            ("write to john.doe@example.com now", "write to <email> now"),
            ("a+tag_1@sub.mail.example.co.uk.", "<email>."),
            ("(bob@example.org), carol@example.net;", "(<email>), <email>;"),
            ("Üser@exämple.com", "<email>"),
            ("two a@b.io c@d.io", "two <email> <email>"),
            ("mailto:me@example.com?subject=hi", "mailto:<email>?subject=hi"),
        ])

    def test_phone_numbers(self):
        self.check([
            ("call +44 20 7946 0958 today", "call <phone> today"),
            ("call +1 (555) 123-4567.", "call <phone>."),
            ("(555) 123-4567", "<phone>"),
            ("555-123-4567", "<phone>"),
            ("555.123.4567", "<phone>"),
            ("555 123 4567", "<phone>"),
            ("+1-555-123-4567", "<phone>"),
            ("+14155552671", "<phone>"),
            ("+49 30 123456", "<phone>"),
            ("06 12 34 56 78", "<phone>"),
            ("tel: 0171 123 4567!", "tel: <phone>!"),
            ("Phone:555-123-4567", "Phone:<phone>"),
        ])

    def test_api_keys_and_tokens(self):
        self.check([
            ("sk-" + "a" * 48, "<secret>"),
            ("key=sk-ant-api03-" + "Ab1-_" * 10, "key=<secret>"),
            ("sk-proj-AbCdEfGhIjKlMnOpQrStUvWx", "<secret>"),
            ("ghp_" + "a1B2c3D4e5" * 4, "<secret>"),
            ("gho_" + "a1B2c3D4e5" * 4, "<secret>"),
            ("github_pat_11ABCDEFG0abcdefghijkl_abcdefghijklmnopqrstuvwxyz0123456789", "<secret>"),
            ("AKIAIOSFODNN7EXAMPLE", "<secret>"),
            ("xoxb-123456789012-abcdefghijkl", "<secret>"),
            ("xoxp-123456789012-abcdefghijkl", "<secret>"),
            ("AIza" + "a1B2c3D4e5" * 3 + "xYz12", "<secret>"),
            ("hf_" + "aB3dE5fG7hI9jK1lM3nO5pQ7rS9tU1vW3x", "<secret>"),
            ("Authorization: Bearer abc123def456ghi789jkl012", "Authorization: Bearer <secret>"),
            ("bearer eyJhbGciOiJIUzI1NiJ9.e30.Et9HFtf9R3GEMA0IICOfFMVXY7kkTX1wr4qCyhIf58U",
             "bearer <secret>"),
            ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0."
             "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U", "<secret>"),
            ("-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\nabc/def+ghi\n-----END RSA PRIVATE KEY----- ok",
             "<secret> ok"),
            ("-----BEGIN PRIVATE KEY-----\nMIIE\n-----END PRIVATE KEY-----", "<secret>"),
        ])

    def test_long_random_strings(self):
        self.check([
            ("d41d8cd98f00b204e9800998ecf8427e", "<secret>"),
            ("sha 3a7bd3e2360a3d29eea436fcfb7e44c735d117c4 here", "sha <secret> here"),
            ("9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08", "<secret>"),
            ("secret: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY.", "secret: <secret>."),
            ("Zm9vYmFyYmF6cXV4MTIzNDU2Nzg5MEFCQ0RFRkdISUo=", "<secret>"),
            ("tok_9fK2mQx7LpVn4RtYb8WcZa3HdJe6GsUo", "<secret>"),
        ])

    def test_ip_addresses(self):
        self.check([
            ("server at 192.168.1.20 and 10.0.0.1:8080", "server at <ip> and <ip>:8080"),
            ("http://8.8.8.8/dns", "http://<ip>/dns"),
            ("255.255.255.255.", "<ip>."),
            ("ping 172.16.254.1, then 1.2.3.4", "ping <ip>, then <ip>"),
            ("loopback 127.0.0.1 and 0.0.0.0 stay", "loopback 127.0.0.1 and 0.0.0.0 stay"),
        ])

    def test_home_folders(self):
        self.check([
            ("C:\\Users\\John Smith\\Documents\\x.txt", "C:\\Users\\<user>\\Documents\\x.txt"),
            ("c:\\users\\bob\\desktop", "c:\\users\\<user>\\desktop"),
            ("D:/Users/Eve/code", "D:/Users/<user>/code"),
            ("/home/alice/projects/app", "/home/<user>/projects/app"),
            ("/Users/bob/Library/x", "/Users/<user>/Library/x"),
            ("cd /home/alice/code && ls /home/bob/", "cd /home/<user>/code && ls /home/<user>/"),
        ])

    def test_card_numbers(self):
        self.check([
            ("4111 1111 1111 1111", "<number>"),
            ("4111-1111-1111-1111", "<number>"),
            ("4111111111111111", "<number>"),
            ("amex 378282246310005!", "amex <number>!"),
            ("5555555555554444", "<number>"),
        ])

    def test_several_kinds_in_one_text(self):
        text = "Email a@b.io, call (555) 123-4567, key sk-" + "x1" * 15 + ", host 10.1.2.3, folder /home/me/x."
        self.assertEqual(redact(text), "Email <email>, call <phone>, key <secret>, host <ip>, folder /home/<user>/x.")

    def test_ordinary_text_is_left_alone(self):
        for text in (
            "",
            "Hello, how are you? I'd like a recipe for pancakes.",
            "Python 3.10.4 and 3.13 are supported; see version 1.2.3 and v2.0.1-beta.",
            "Released on 2024-01-15 at 10:30:45, ISO 2024-01-15T10:30:45Z, 15.01.2024 and 01-02-2024.",
            "Compute 1234 * 5678 = 7006652 and 2**64 = 18446744073709551616.",
            "The number 1,234,567.89 and 0.5 1 2 3.",
            "Timestamp 1700000000000 and id 4111111111111112 and 1234567890123456.",
            "uuid 123e4567-e89b-12d3-a456-426614174000 and short sha 3a7bd3e and id a1b2c3d4.",
            "GET /home/dashboard and src/home/pages/index.js and ./home/alice.txt",
            "C:\\Users\\Public\\Documents and C:\\Users\\Default\\x",
            "class TestUserLoginWithExpiredTokenReturns401Error: pass",
            "def test_user_login_with_expired_token_returns_401(): pass",
            "ThisIsAVeryLongCamelCaseFunctionNameForSomething and MAX_CONNECTIONS_PER_USER_2024_LIMIT_VALUE",
            "/usr/lib/python3/dist-packages/foo/bar/baz/some_module_name.py",
            "src/components/UserProfile/UserProfileHeader2.tsx and Tools/build/parse_html5_entities",
            "https://example.com/Docs2/GettingStarted/InstallationGuide?lang=en",
            "@property\ndef x(self): return a @ b",
            "Bearer of bad news; bearer tokens are explained below; a task-based risk-assessment of sk-learn.",
            "risk-free desk-lamp task-management-system-overview-for-everyone-in-the-team",
            "phone numbers look like 555-1234 or 12-34-56 or 123456",
            "5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100",
            "x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; y = 3.14159265358979323846",
            "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz and aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "The path is C:\\Program Files\\App\\bin and /usr/local/bin and ~/projects/x",
        ):
            with self.subTest(text=text):
                self.assertEqual(redact(text), text)

    def test_redaction_is_idempotent(self):
        text = ("bob@example.com +44 20 7946 0958 sk-" + "ab12" * 8 + " 192.168.0.4 C:\\Users\\Bob\\x /home/bob/y "
                "4111 1111 1111 1111 d41d8cd98f00b204e9800998ecf8427e Bearer abc123def456ghi789jkl012")
        once = redact(text)
        self.assertEqual(redact(once), once)
        for placeholder in ("<email>", "<phone>", "<secret>", "<ip>", "<user>", "<number>"):
            self.assertIn(placeholder, once)

    def test_large_hostile_inputs_finish_quickly(self):
        hostile = ["1 " * 100_000, "1-" * 100_000, "+1" * 100_000, "a@" * 100_000, "a." * 100_000, "A/b" * 70_000,
                   "-----BEGIN PRIVATE KEY-----\n" * 7_000, "1.1.1." * 30_000, "x" * 70 + "@a" * 60_000,
                   "(555) " * 30_000, "ab12" * 50_000, "/home/" * 33_000, "C:\\Users\\" * 22_000, "Ab1" * 70_000]
        started = time.monotonic()
        for text in hostile:
            redact(text)
        self.assertLess(time.monotonic() - started, 30)


# ------------------------------------------------------------------------------------------------ the bundle
class TrainingBundleTests(unittest.TestCase):
    def test_script_is_valid_python(self):
        source = train_script()
        compile(source, "train_lora.py", "exec")
        tree = ast.parse(source)
        self.assertIn("main", [n.name for n in tree.body if isinstance(n, ast.FunctionDef)])
        self.assertTrue(source.startswith("#!/usr/bin/env python3"))

    def test_script_states_what_it_is_and_how_to_use_the_result(self):
        source = train_script()
        for needle in ("TEMPLATE", "NOT been run", "openai/gpt-oss-20b", "Mxfp4Config", "dequantize=True", "all-linear",
                       "LORA_RANK = 16", "EPOCHS = 3", "LEARNING_RATE = 2e-4", '"cosine"', "packing=False", "sft.jsonl",
                       "convert_lora_to_gguf.py", "LoRA adapter (.gguf)", "48-80 GB", "gpt-oss-120b"):
            self.assertIn(needle, source)

    def test_readme_is_honest_and_practical(self):
        text = training_readme()
        for needle in ("sft.jsonl", "pref.jsonl", "manifest.json", "train_lora.py", "privacy", "best effort",
                       "LoRA adapter (.gguf)", "gpt-oss-20b", "not** on gpt-oss-120b", "unmeasured"):
            self.assertIn(needle, text)

    def run_script(self, examples):
        """Execute the script's main() against stub ML libraries (this checks wiring, not the real APIs)."""
        with tempfile.TemporaryDirectory() as tmp:
            if examples is not None:
                (Path(tmp) / "sft.jsonl").write_text("\n".join(examples) + "\n", encoding="utf-8")
            torch = mock.MagicMock()
            torch.cuda.is_available.return_value = True
            torch.cuda.device_count.return_value = 1
            torch.cuda.get_device_properties.return_value.total_memory = 80 * 2**30
            torch.cuda.get_device_name.return_value = "Fake GPU"
            modules = {name: mock.MagicMock() for name in ("datasets", "peft", "transformers", "trl")}
            modules["torch"] = torch
            namespace = {"__name__": "train_lora", "__file__": str(Path(tmp) / "train_lora.py")}
            exec(compile(train_script(), "train_lora.py", "exec"), namespace)
            with mock.patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
                namespace["main"]()
            return modules, namespace, tmp

    def chat(self, i):
        return json.dumps({"messages": [{"role": "user", "content": f"Question {i}?"},
                                        {"role": "assistant", "content": f"Answer {i}."}]})

    def test_script_wiring_with_stub_libraries(self):
        modules, _, tmp = self.run_script([self.chat(i) for i in range(25)])
        modules["transformers"].Mxfp4Config.assert_called_once_with(dequantize=True)
        load = modules["transformers"].AutoModelForCausalLM.from_pretrained
        self.assertEqual(load.call_args.args, ("openai/gpt-oss-20b",))
        self.assertIs(load.call_args.kwargs["quantization_config"], modules["transformers"].Mxfp4Config.return_value)
        lora = modules["peft"].LoraConfig.call_args.kwargs
        self.assertEqual((lora["r"], lora["target_modules"]), (16, "all-linear"))
        config = modules["trl"].SFTConfig.call_args.kwargs
        self.assertEqual((config["num_train_epochs"], config["learning_rate"], config["lr_scheduler_type"],
                          config["packing"], config["bf16"]), (3, 2e-4, "cosine", False, True))
        dataset_rows = modules["datasets"].Dataset.from_list.call_args.args[0]
        self.assertEqual(len(dataset_rows), 25)
        self.assertEqual(dataset_rows[0]["messages"][-1]["role"], "assistant")
        trainer = modules["trl"].SFTTrainer.return_value
        trainer.train.assert_called_once_with()
        trainer.save_model.assert_called_once_with(str(Path(tmp).resolve() / "duckstoken-lora"))

    def test_script_stops_early_with_clear_messages(self):
        with self.assertRaisesRegex(SystemExit, "sft.jsonl"):
            self.run_script(None)
        with self.assertRaisesRegex(SystemExit, "Only 3 usable examples"):
            self.run_script([self.chat(i) for i in range(3)])
        with self.assertRaisesRegex(SystemExit, "line 2"):
            self.run_script([self.chat(0), "not json"])
        with self.assertRaisesRegex(SystemExit, "line 1"):
            self.run_script(['{"messages": []}'])

    def test_script_skips_chats_without_a_final_answer_and_blank_lines(self):
        unfinished = json.dumps({"messages": [{"role": "user", "content": "Hi?"}]})
        modules, _, _ = self.run_script([self.chat(i) for i in range(20)] + ["", unfinished])
        self.assertEqual(len(modules["datasets"].Dataset.from_list.call_args.args[0]), 20)

    def test_script_refuses_to_run_without_a_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "sft.jsonl").write_text("\n".join(self.chat(i) for i in range(20)), encoding="utf-8")
            torch = mock.MagicMock()
            torch.cuda.is_available.return_value = False
            modules = {name: mock.MagicMock() for name in ("datasets", "peft", "transformers", "trl")}
            modules["torch"] = torch
            namespace = {"__name__": "train_lora", "__file__": str(Path(tmp) / "train_lora.py")}
            exec(compile(train_script(), "train_lora.py", "exec"), namespace)
            with mock.patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(SystemExit, "No CUDA GPU"):
                    namespace["main"]()
            modules["transformers"].AutoModelForCausalLM.from_pretrained.assert_not_called()


if __name__ == "__main__":
    unittest.main()
