import json
import os
import random
import shutil
import sqlite3
import string
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["DUCKSTOKEN_HOME"] = tempfile.mkdtemp(prefix="duck-test-memory-")

from duckstoken import memory  # noqa: E402
from duckstoken.db import Database  # noqa: E402
from duckstoken.memory import MemoryStore  # noqa: E402

ITEM_KEYS = {"id", "text", "kind", "source", "url", "title", "pinned", "enabled", "uses", "created", "updated"}


def _probe() -> bool:
    tmp = tempfile.mkdtemp(prefix="duck-test-fts-")
    try:
        db = Database(Path(tmp) / "probe.db")
        has = db.has_fts
        db.conn().close()
        return has
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


HAS_FTS = _probe()


class StoreCase(unittest.TestCase):
    """Fresh database and store for every test."""

    def setUp(self):
        tmp = tempfile.mkdtemp(prefix="duck-test-memory-db-")
        self.db = Database(Path(tmp) / "x.db")
        self.store = MemoryStore(self.db)
        self.addCleanup(shutil.rmtree, tmp, True)
        self.addCleanup(lambda: self.db.conn().close())

    def texts(self, results):
        return [r["text"] for r in results]


class CrudTests(StoreCase):
    def test_kinds_and_defaults(self):
        self.assertEqual(memory.KINDS, ("fact", "preference", "project", "lesson", "note", "web", "doc"))
        item = self.store.add("I drink green tea")
        self.assertEqual(set(item), ITEM_KEYS)
        self.assertEqual((item["kind"], item["source"], item["url"], item["title"]), ("fact", "user", "", ""))
        self.assertIs(item["pinned"], False)
        self.assertIs(item["enabled"], True)
        self.assertEqual(item["uses"], 0)
        self.assertEqual(item["created"], item["updated"])
        self.assertEqual(self.store.get(item["id"]), item)

    def test_add_with_all_fields(self):
        item = self.store.add("Use tabs", kind="preference", source="auto", url="https://example.com/a?b=1",
                              title="  Style   guide ", pinned=True)
        self.assertEqual((item["kind"], item["source"], item["title"], item["pinned"]),
                         ("preference", "auto", "Style guide", True))
        self.assertEqual(item["url"], "https://example.com/a?b=1")

    def test_text_is_stripped_and_control_characters_removed(self):
        self.assertEqual(self.store.add("  hello \x00world\x07 \n")["text"], "hello world")
        self.assertEqual(self.store.add("two\r\nlines\twith tab")["text"], "two\nlines\twith tab")

    def test_rejects_bad_text(self):
        for bad in ("", "   \n", "\x00\x01", None, 123, ["a"], "x" * 4001):
            with self.assertRaises(ValueError, msg=repr(bad)[:30]):
                self.store.add(bad)
        self.assertEqual(len(self.store.add("x" * 4000)["text"]), 4000)
        self.assertEqual(self.store.stats()["total"], 1)

    def test_rejects_bad_metadata(self):
        for kwargs in ({"kind": "bogus"}, {"kind": None}, {"kind": ""}, {"source": "nobody"},
                       {"url": "javascript:alert(1)"}, {"url": "ftp://example.com/x"}, {"url": "http://"},
                       {"url": "https://exa mple.com"}, {"url": "https://a.b/" + "a" * 2001}, {"url": 5},
                       {"title": 5}, {"pinned": "yes"}, {"pinned": 2}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                self.store.add("some text", **kwargs)
        self.assertEqual(self.store.stats()["total"], 0)

    def test_title_is_trimmed_not_rejected(self):
        self.assertEqual(len(self.store.add("t", title="w" * 1000)["title"]), memory.MAX_TITLE)

    def test_dedupe_ignores_case_and_whitespace(self):
        first = self.store.add("I like   Green Tea")
        with mock.patch.object(memory, "_now", return_value=first["created"] + 100):
            again = self.store.add("  i like green\ttea ")
        self.assertEqual(again["id"], first["id"])
        self.assertEqual(again["text"], "I like   Green Tea")
        self.assertEqual(again["updated"], first["created"] + 100)
        self.assertEqual(again["created"], first["created"])
        self.assertEqual(self.store.stats()["total"], 1)

    def test_dedupe_is_per_kind(self):
        a = self.store.add("same words", kind="fact")
        b = self.store.add("same words", kind="note")
        self.assertNotEqual(a["id"], b["id"])
        self.assertEqual(self.store.stats()["total"], 2)

    def test_dedupe_unicode_casefold(self):
        a = self.store.add("Straße 5")
        self.assertEqual(self.store.add("STRASSE 5")["id"], a["id"])

    def test_dedupe_can_pin_but_never_unpins(self):
        a = self.store.add("remember this")
        self.assertTrue(self.store.add("remember this", pinned=True)["pinned"])
        self.assertTrue(self.store.add("remember this", pinned=False)["pinned"])
        self.assertEqual(self.store.stats()["pinned"], 1)
        self.assertEqual(self.store.get(a["id"])["pinned"], True)

    def test_get_with_odd_ids(self):
        item = self.store.add("hello there")
        self.assertEqual(self.store.get(str(item["id"]))["id"], item["id"])
        for bad in (None, "abc", "", -1, 0, 2 ** 70, True, 1.5, [1], {"id": 1}, 999):
            self.assertIsNone(self.store.get(bad), repr(bad))

    def test_update_each_field(self):
        item = self.store.add("old text", title="t")
        with mock.patch.object(memory, "_now", return_value=item["created"] + 50):
            upd = self.store.update(item["id"], text=" new text ", kind="project", pinned=True, enabled=False,
                                    title="New title", url="https://example.org/")
        self.assertEqual((upd["text"], upd["kind"], upd["pinned"], upd["enabled"], upd["title"], upd["url"]),
                         ("new text", "project", True, False, "New title", "https://example.org/"))
        self.assertEqual(upd["updated"], item["created"] + 50)
        self.assertEqual(upd["created"], item["created"])
        self.assertEqual(self.store.update(str(item["id"]), pinned=0)["pinned"], False)
        self.assertEqual(self.store.update(item["id"], url="")["url"], "")

    def test_update_validates(self):
        item = self.store.add("keep me")
        for kwargs in ({"text": ""}, {"text": "x" * 4001}, {"kind": "nope"}, {"pinned": "true"},
                       {"enabled": None}, {"url": "javascript:1"}, {"uses": 5}, {"created": 1.0}, {"source": "user"}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                self.store.update(item["id"], **kwargs)
        self.assertEqual(self.store.get(item["id"])["text"], "keep me")

    def test_update_missing_item_and_noop(self):
        self.assertIsNone(self.store.update(12345, text="x"))
        self.assertIsNone(self.store.update("junk", text="x"))
        item = self.store.add("unchanged")
        self.assertEqual(self.store.update(item["id"]), item)

    def test_update_cannot_create_duplicates(self):
        a = self.store.add("alpha fact")
        b = self.store.add("beta fact")
        with self.assertRaises(ValueError):
            self.store.update(b["id"], text="  ALPHA   fact ")
        self.assertEqual(self.store.get(b["id"])["text"], "beta fact")
        self.assertEqual(self.store.update(a["id"], text="Alpha Fact")["text"], "Alpha Fact")
        note = self.store.add("alpha fact", kind="note")
        with self.assertRaises(ValueError):
            self.store.update(note["id"], kind="fact")

    def test_delete(self):
        item = self.store.add("temporary")
        self.assertTrue(self.store.delete(item["id"]))
        self.assertFalse(self.store.delete(item["id"]))
        self.assertFalse(self.store.delete("nope"))
        self.assertIsNone(self.store.get(item["id"]))

    def test_clear(self):
        self.store.add("a note", kind="note")
        self.store.add("another note", kind="note")
        self.store.add("a fact")
        self.assertEqual(self.store.clear("note"), 2)
        self.assertEqual(self.store.stats()["total"], 1)
        with self.assertRaises(ValueError):
            self.store.clear("bogus")
        with self.assertRaises(ValueError):
            self.store.clear("")
        self.assertEqual(self.store.clear(), 1)
        self.assertEqual(self.store.clear(), 0)

    def test_sql_metacharacters_are_just_data(self):
        nasty = "'; DROP TABLE memories; -- \" ) OR 1=1"
        item = self.store.add(nasty, title=nasty)
        self.assertEqual(self.store.get(item["id"])["text"], nasty)
        self.assertEqual(self.store.list(q=nasty)["total"], 1)
        self.assertEqual(self.store.list(kind=nasty)["total"], 0)
        self.assertEqual(self.store.list(q="x' OR '1'='1")["total"], 0)
        self.store.search(nasty)
        self.assertEqual(self.store.stats()["total"], 1)


class ListTests(StoreCase):
    def test_newest_first_with_paging(self):
        for i in range(7):
            with mock.patch.object(memory, "_now", return_value=1000.0 + i):
                self.store.add(f"memory number {i}")
        page = self.store.list(limit=3)
        self.assertEqual(page["total"], 7)
        self.assertEqual(self.texts(page["items"]), ["memory number 6", "memory number 5", "memory number 4"])
        self.assertEqual(self.texts(self.store.list(limit=3, offset=3)["items"]),
                         ["memory number 3", "memory number 2", "memory number 1"])
        self.assertEqual(self.texts(self.store.list(limit=3, offset=6)["items"]), ["memory number 0"])
        self.assertEqual(self.store.list(limit=3, offset=50)["items"], [])
        self.assertTrue(all(set(i) == ITEM_KEYS for i in page["items"]))

    def test_kind_and_text_filters(self):
        self.store.add("blue sky", kind="note")
        self.store.add("blue whale", kind="fact")
        self.store.add("red apple", kind="fact", title="Fruit about BLUE things")
        self.assertEqual(self.store.list(kind="fact")["total"], 2)
        self.assertEqual(self.store.list(q="blue")["total"], 3)
        self.assertEqual(self.store.list(q="BLUE", kind="fact")["total"], 2)
        self.assertEqual(self.store.list(q="  whale ")["total"], 1)
        self.assertEqual(self.store.list(kind="bogus")["total"], 0)

    def test_like_wildcards_are_literal(self):
        for t in ("100% sure", "snake_case names", "snakeXcase names", "50 percent", "C:\\temp\\file", "plain"):
            self.store.add(t)
        self.assertEqual(self.texts(self.store.list(q="%")["items"]), ["100% sure"])
        self.assertEqual(self.texts(self.store.list(q="_")["items"]), ["snake_case names"])
        self.assertEqual(self.texts(self.store.list(q="snake_case")["items"]), ["snake_case names"])
        self.assertEqual(self.texts(self.store.list(q="\\temp")["items"]), ["C:\\temp\\file"])
        self.assertEqual(self.store.list(q="\\")["total"], 1)
        self.assertEqual(self.store.list(q="%%")["total"], 0)

    def test_odd_arguments_are_clamped(self):
        for i in range(3):
            self.store.add(f"item {i}")
        self.assertEqual(len(self.store.list(limit=0)["items"]), 1)
        self.assertEqual(len(self.store.list(limit=-5)["items"]), 1)
        self.assertEqual(len(self.store.list(limit=10 ** 9)["items"]), 3)
        self.assertEqual(len(self.store.list(limit="2")["items"]), 2)
        self.assertEqual(len(self.store.list(limit="junk")["items"]), 3)
        self.assertEqual(len(self.store.list(offset=-4)["items"]), 3)
        self.assertEqual(self.store.list(q=None)["total"], 3)
        self.assertEqual(self.store.list(q=["a"])["total"], 3)


class SearchBehaviour:
    """Search tests that must hold for the FTS5 backend and for the pure-Python fallback."""

    def add_many(self, *texts, **kwargs):
        return [self.store.add(t, **kwargs) for t in texts]

    def test_most_relevant_first(self):
        self.add_many("I drink green tea every morning", "My favourite editor is Neovim",
                      "Notes on the tea ceremony: matcha whisk and bowl", "The server rack is in the garage")
        results = self.store.search("which tea do I drink?")
        self.assertEqual(results[0]["text"], "I drink green tea every morning")
        self.assertEqual(len(results), 2)
        self.assertEqual(self.texts(self.store.search("editor")), ["My favourite editor is Neovim"])

    def test_results_carry_scores_in_descending_order(self):
        self.add_many("tea tea tea", "tea and coffee", "coffee only", "tea")
        results = self.store.search("tea")
        self.assertEqual(len(results), 3)
        self.assertTrue(all(set(r) == ITEM_KEYS | {"score"} for r in results))
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(s > 0 for s in scores))

    def test_unrelated_query_finds_nothing(self):
        self.add_many("I drink green tea", "I use Neovim")
        self.assertEqual(self.store.search("quantum chromodynamics"), [])

    def test_rarer_words_weigh_more(self):
        self.add_many("deploy staging checklist", "deploy production checklist", "deploy rollback checklist",
                      "deploy notes", "deploy window friday", "kubernetes cluster upgrade")
        self.assertEqual(self.store.search("deploy kubernetes", k=1)[0]["text"], "kubernetes cluster upgrade")

    def test_more_matching_words_rank_higher(self):
        self.add_many("rust game engine", "rust", "game", "rust game")
        self.assertEqual(self.store.search("rust game engine", k=1)[0]["text"], "rust game engine")
        self.assertEqual(self.texts(self.store.search("rust game"))[:2], ["rust game", "rust game engine"])

    def test_title_is_searched(self):
        self.store.add("Pods, services and ingress", title="Kubernetes handbook", kind="doc")
        self.assertEqual(self.texts(self.store.search("kubernetes")), ["Pods, services and ingress"])

    def test_plural_forms_match(self):
        self.add_many("I ride bikes on weekends")
        self.assertEqual(len(self.store.search("bike")), 1)
        self.assertEqual(len(self.store.search("bikes")), 1)

    def test_unicode_text(self):
        self.add_many("Café Müller liegt in München", "日本語のメモ")
        self.assertEqual(self.texts(self.store.search("München")), ["Café Müller liegt in München"])
        self.assertEqual(self.texts(self.store.search("müller")), ["Café Müller liegt in München"])
        self.assertEqual(self.texts(self.store.search("日本語のメモ")), ["日本語のメモ"])

    def test_disabled_items_are_never_returned(self):
        a, b = self.add_many("alpha topic", "alpha subject")
        self.store.update(a["id"], enabled=False, pinned=True)
        self.assertEqual(self.texts(self.store.search("alpha")), ["alpha subject"])
        self.assertEqual(self.texts(self.store.search("")), [])

    def test_kinds_filter(self):
        self.store.add("tea in a note", kind="note")
        self.store.add("tea as a fact", kind="fact")
        self.store.add("tea as a preference", kind="preference", pinned=True)
        self.assertEqual(len(self.store.search("tea")), 3)
        self.assertEqual(self.texts(self.store.search("tea", kinds=["note"])), ["tea in a note"])
        self.assertEqual(self.texts(self.store.search("tea", kinds="fact")), ["tea as a fact"])
        self.assertEqual(len(self.store.search("tea", kinds=("note", "fact"))), 2)
        self.assertEqual(self.store.search("tea", kinds=[]), [])
        self.assertEqual(self.store.search("tea", kinds=["bogus"]), [])
        self.assertEqual(self.texts(self.store.search("tea", kinds=["bogus", "note", ["unhashable"]])),
                         ["tea in a note"])

    def test_k_limits_results(self):
        self.add_many(*[f"tea fact {i}" for i in range(10)])
        self.assertEqual(len(self.store.search("tea")), 6)
        self.assertEqual(len(self.store.search("tea", k=3)), 3)
        self.assertEqual(len(self.store.search("tea", k=50)), 10)
        self.assertEqual(self.store.search("tea", k=0), [])
        self.assertEqual(self.store.search("tea", k=-2), [])
        self.assertEqual(len(self.store.search("tea", k="2")), 2)

    def test_pinned_items_come_first_even_without_a_match(self):
        self.add_many("I drink green tea", "Tea is grown in Assam")
        pin = self.store.add("My name is Sam", pinned=True)
        results = self.store.search("tea")
        self.assertEqual(results[0]["id"], pin["id"])
        self.assertEqual(results[0]["score"], 0.0)
        self.assertEqual(len(results), 3)
        self.assertEqual(self.texts(self.store.search("something else entirely")), ["My name is Sam"])
        self.assertEqual(self.texts(self.store.search("")), ["My name is Sam"])

    def test_pinned_items_count_towards_k(self):
        self.add_many("tea one", "tea two", "tea three")
        pins = [self.store.add(f"pinned note {i}", pinned=True) for i in range(2)]
        results = self.store.search("tea", k=3)
        self.assertEqual(len(results), 3)
        self.assertEqual({r["id"] for r in results[:2]}, {p["id"] for p in pins})
        self.assertEqual(len(self.store.search("tea", k=2)), 2)
        self.assertTrue(all(r["pinned"] for r in self.store.search("tea", k=2)))
        self.assertTrue(all(r["pinned"] for r in self.store.search("tea", k=1)))

    def test_pinned_items_are_ordered_by_relevance(self):
        irrelevant = self.store.add("unrelated pinned thing", pinned=True)
        relevant = self.store.add("pinned tea preference", pinned=True)
        results = self.store.search("tea")
        self.assertEqual([r["id"] for r in results], [relevant["id"], irrelevant["id"]])

    def test_stopword_only_queries_match_only_pinned(self):
        self.add_many("the quick brown fox", "what is this")
        pin = self.store.add("always remembered", pinned=True)
        for q in ("the and of", "what is this", "a", "i", ""):
            self.assertEqual([r["id"] for r in self.store.search(q)], [pin["id"]], q)

    def test_hostile_queries_never_raise(self):
        self.add_many("tea time near the house", "and or not", "quote \" inside", "star * wildcard", "col: value",
                      "emoji 🙂 here")
        hostile = ['"', "'", "*", "**", "OR", "AND", "NOT", "NEAR(", "NEAR(a b)", ")", "(", "()", "-", "--", "^",
                   ":", "text:tea", "title: x", '"unterminated', 'tea" OR "', "tea OR", "OR tea", "a AND", "{x}",
                   "[x]", "tea*", "-tea", "+tea", "NEAR/3", "🙂", "🙂🙂🙂", "tea 🙂", "\x00", "tea\x00time",
                   "\\",
                   "'; DROP TABLE memories; --", "%", "_", "ａｂｃ", "\u202e", "\n\t ", "x" * 100_000,
                   "tea " * 50_000, "OR " * 5000, None, 5, 5.5, [], {}, b"tea", ["tea"]]
        for q in hostile:
            with self.subTest(query=repr(q)[:40]):
                self.assertIsInstance(self.store.search(q), list)
        self.assertEqual(self.store.stats()["total"], 6)
        self.assertEqual(self.texts(self.store.search("tea OR")), ["tea time near the house"])
        self.assertEqual(self.texts(self.store.search('"tea"')), ["tea time near the house"])

    def test_operator_words_are_plain_words(self):
        self.add_many("tea time", "coffee time")
        self.assertEqual(len(self.store.search("tea NOT coffee")), 2)
        self.assertEqual(len(self.store.search("NEAR(tea coffee)")), 2)

    def test_query_with_many_words_is_bounded(self):
        self.add_many("needle", "haystack")
        query = " ".join(f"word{i}" for i in range(500)) + " needle"
        self.assertIsInstance(self.store.search(query), list)

    def test_index_follows_updates_and_deletes(self):
        item, = self.add_many("I own a red bicycle")
        self.assertEqual(len(self.store.search("bicycle")), 1)
        self.store.update(item["id"], text="I own a blue scooter")
        self.assertEqual(self.store.search("bicycle"), [])
        self.assertEqual(len(self.store.search("scooter")), 1)
        self.store.update(item["id"], title="Vehicles")
        self.assertEqual(len(self.store.search("vehicles")), 1)
        self.store.delete(item["id"])
        self.assertEqual(self.store.search("scooter"), [])
        self.store.add("temporary tea")
        self.store.clear()
        self.assertEqual(self.store.search("tea"), [])

    def test_search_does_not_count_as_use(self):
        item, = self.add_many("tea")
        self.store.search("tea")
        self.assertEqual(self.store.get(item["id"])["uses"], 0)

    def test_documents_are_searchable(self):
        paragraphs = [f"Chapter {i} explains topic{i} in detail. " * 12 for i in range(8)]
        self.assertGreater(self.store.add_document("Handbook", "\n\n".join(paragraphs), "https://example.com/h"), 1)
        hit = self.store.search("topic5")
        self.assertTrue(hit)
        self.assertEqual((hit[0]["kind"], hit[0]["title"], hit[0]["url"]), ("doc", "Handbook", "https://example.com/h"))
        self.assertIn("topic5", hit[0]["text"])
        self.assertTrue(self.store.search("handbook"))

    def test_build_context_works_with_this_backend(self):
        self.add_many("I drink green tea", "I use Neovim")
        ctx = self.store.build_context("tea?", 3, 500)
        self.assertEqual(ctx["text"], memory.CONTEXT_HEADER + "\n- I drink green tea")


@unittest.skipUnless(HAS_FTS, "this Python's SQLite has no FTS5")
class FtsSearchTests(SearchBehaviour, StoreCase):
    def setUp(self):
        super().setUp()
        self.assertTrue(self.db.has_fts)

    def test_python_ranker_is_not_used(self):
        self.add_many("I drink green tea")
        with mock.patch.object(self.store, "_match_python", side_effect=AssertionError("fallback used")):
            self.assertEqual(len(self.store.search("tea")), 1)

    def test_fts_failure_degrades_to_pinned_only(self):
        self.add_many("I drink green tea")
        pin = self.store.add("pinned thing", pinned=True)
        real_query = self.db.query

        def flaky(sql, params=()):
            if "MATCH" in sql:
                raise sqlite3.OperationalError("fts5: syntax error")
            return real_query(sql, params)

        with mock.patch.object(self.db, "query", side_effect=flaky):
            self.assertEqual([r["id"] for r in self.store.search("tea")], [pin["id"]])

    def test_stats_report_fts(self):
        self.assertTrue(self.store.stats()["fts"])


class PythonSearchTests(SearchBehaviour, StoreCase):
    def setUp(self):
        super().setUp()
        self.db.has_fts = False

    def test_fts_is_not_used(self):
        self.add_many("I drink green tea")
        with mock.patch.object(self.store, "_match_fts", side_effect=AssertionError("fts used")):
            self.assertEqual(len(self.store.search("tea")), 1)

    def test_stats_report_no_fts(self):
        self.assertFalse(self.store.stats()["fts"])


class BuildContextTests(StoreCase):
    def test_format_and_bookkeeping(self):
        a = self.store.add("I drink green tea")
        b = self.store.add("Tea is grown in Assam")
        self.store.add("I use Neovim")
        ctx = self.store.build_context("tea", 5, 1000)
        self.assertTrue(ctx["text"].startswith(memory.CONTEXT_HEADER + "\n- "))
        self.assertEqual(ctx["text"].count("\n- "), 2)
        self.assertEqual(sorted(ctx["ids"]), sorted([a["id"], b["id"]]))
        self.assertEqual([i["id"] for i in ctx["items"]], ctx["ids"])
        self.assertTrue(all(i["uses"] == 1 for i in ctx["items"]))
        self.assertEqual(self.store.get(a["id"])["uses"], 1)
        self.assertEqual(self.store.get(b["id"])["uses"], 1)
        self.assertEqual(self.store.build_context("tea", 5, 1000)["items"][0]["uses"], 2)
        untouched = [i for i in self.store.list()["items"] if i["text"] == "I use Neovim"][0]
        self.assertEqual(untouched["uses"], 0)

    def test_nothing_relevant_gives_empty_block(self):
        item = self.store.add("I drink green tea")
        for q in ("quantum physics", "", "the of and"):
            self.assertEqual(self.store.build_context(q, 5, 1000), {"text": "", "ids": [], "items": []})
        self.assertEqual(self.store.get(item["id"])["uses"], 0)

    def test_empty_store(self):
        self.assertEqual(self.store.build_context("anything", 5, 1000), {"text": "", "ids": [], "items": []})

    def test_pinned_items_are_always_included(self):
        pin = self.store.add("Name is Sam", pinned=True)
        ctx = self.store.build_context("unrelated question", 3, 500)
        self.assertEqual(ctx["ids"], [pin["id"]])
        self.assertIn("- Name is Sam", ctx["text"])

    def test_max_items(self):
        for i in range(8):
            self.store.add(f"tea fact {i}")
        self.assertEqual(len(self.store.build_context("tea", 3, 5000)["ids"]), 3)
        self.assertEqual(len(self.store.build_context("tea", 1, 5000)["ids"]), 1)
        for bad in (0, -1, None, "x"):
            self.assertEqual(self.store.build_context("tea", bad, 5000)["text"], "")

    def test_max_chars_is_never_exceeded(self):
        for i in range(12):
            self.store.add(f"tea note {i}: " + "word " * (5 + 40 * (i % 4)))
        self.store.add("pinned background " + "info " * 60, pinned=True)
        self.store.add("tea " * 400)
        for budget in list(range(0, 400, 7)) + [500, 777, 1000, 1800]:
            with self.subTest(budget=budget):
                ctx = self.store.build_context("tea", 10, budget)
                self.assertLessEqual(len(ctx["text"]), budget)
                if ctx["text"]:
                    self.assertTrue(ctx["text"].startswith(memory.CONTEXT_HEADER))
                    self.assertEqual(ctx["text"].count("\n- "), len(ctx["ids"]))
                    self.assertEqual(len(ctx["items"]), len(ctx["ids"]))
                else:
                    self.assertEqual((ctx["ids"], ctx["items"]), ([], []))

    def test_budget_too_small_for_header_gives_nothing(self):
        self.store.add("tea", pinned=True)
        self.assertEqual(self.store.build_context("tea", 5, len(memory.CONTEXT_HEADER))["text"], "")
        self.assertEqual(self.store.get(1)["uses"], 0)

    def test_long_item_is_truncated_with_an_ellipsis(self):
        self.store.add("tea " + "very long story " * 200)
        ctx = self.store.build_context("tea", 3, 400)
        line = ctx["text"].split("\n")[1]
        self.assertTrue(line.endswith(memory.ELLIPSIS))
        self.assertLessEqual(len(ctx["text"]), 400)
        self.assertTrue(line.startswith("- tea very long story"))

    def test_one_long_item_cannot_crowd_out_the_rest(self):
        self.store.add("tea " + "long " * 600)
        self.store.add("tea is relaxing")
        ctx = self.store.build_context("tea", 3, 1000)
        self.assertEqual(len(ctx["ids"]), 2)
        self.assertIn("- tea is relaxing", ctx["text"])

    def test_whitespace_is_collapsed_to_single_lines(self):
        self.store.add("tea\n\nis   good\n- really")
        text = self.store.build_context("tea", 3, 500)["text"]
        self.assertEqual(text.split("\n")[1:], ["- tea is good - really"])

    def test_titles_are_shown_for_documents_and_web_notes(self):
        self.store.add("Pods run containers", kind="doc", title="K8s guide")
        self.store.add("Pods are cute", kind="note", title="ignored title")
        text = self.store.build_context("pods", 5, 500)["text"]
        self.assertIn("- [K8s guide] Pods run containers", text)
        self.assertIn("- Pods are cute", text)

    def test_disabled_items_are_not_included(self):
        item = self.store.add("tea", pinned=True)
        self.store.update(item["id"], enabled=False)
        self.assertEqual(self.store.build_context("tea", 3, 500)["text"], "")

    def test_works_without_fts(self):
        self.db.has_fts = False
        self.store.add("I drink green tea")
        ctx = self.store.build_context("tea", 3, 500)
        self.assertEqual(len(ctx["ids"]), 1)


class StatsTests(StoreCase):
    def test_empty(self):
        stats = self.store.stats()
        self.assertEqual((stats["total"], stats["pinned"]), (0, 0))
        self.assertEqual(stats["by_kind"], dict.fromkeys(memory.KINDS, 0))
        self.assertEqual(stats["fts"], self.db.has_fts)

    def test_counts(self):
        self.store.add("a", kind="fact", pinned=True)
        self.store.add("b", kind="fact")
        self.store.add("c", kind="note", pinned=True)
        self.store.add("d", kind="doc")
        stats = self.store.stats()
        self.assertEqual(stats["total"], 4)
        self.assertEqual(stats["pinned"], 2)
        self.assertEqual((stats["by_kind"]["fact"], stats["by_kind"]["note"], stats["by_kind"]["doc"]), (2, 1, 1))
        self.assertEqual(sum(stats["by_kind"].values()), 4)


class ImportExportTests(StoreCase):
    def fresh_store(self):
        tmp = tempfile.mkdtemp(prefix="duck-test-memory-db-")
        db = Database(Path(tmp) / "y.db")
        self.addCleanup(shutil.rmtree, tmp, True)
        self.addCleanup(lambda: db.conn().close())
        return MemoryStore(db)

    def test_round_trip_preserves_everything_but_ids(self):
        self.store.add("I drink green tea", kind="preference", source="auto", pinned=True)
        self.store.add("Docs page", kind="web", source="web", url="https://example.com/p", title="Example")
        off = self.store.add("disabled thing", kind="note")
        self.store.update(off["id"], enabled=False)
        owls, foxes = "First paragraph about owls. " * 4, "Second paragraph about foxes. " * 4
        self.assertEqual(self.store.add_document("Handbook", f"{owls}\n\n{foxes}", chunk_chars=200, overlap=0), 2)
        self.store.build_context("tea", 3, 500)
        exported = json.loads(json.dumps(self.store.export_all()))
        self.assertEqual(len(exported), 5)
        self.assertEqual(exported, self.store.export_all())

        other = self.fresh_store()
        self.assertEqual(other.import_items(exported), 5)
        without_ids = [{k: v for k, v in i.items() if k != "id"} for i in exported]
        self.assertEqual([{k: v for k, v in i.items() if k != "id"} for i in other.export_all()], without_ids)
        self.assertEqual(other.stats()["total"], 5)
        self.assertEqual(self.texts(other.search("tea")), ["I drink green tea"])

    def test_export_is_oldest_first(self):
        for i in range(3):
            self.store.add(f"item {i}")
        self.assertEqual(self.texts(self.store.export_all()), ["item 0", "item 1", "item 2"])

    def test_import_is_idempotent_and_dedupes(self):
        items = [{"text": "alpha"}, {"text": "  ALPHA "}, {"text": "beta"}, {"text": "alpha", "kind": "note"}]
        self.assertEqual(self.store.import_items(items), 3)
        self.assertEqual(self.store.import_items(items), 0)
        self.assertEqual(self.store.import_items([{"text": "beta"}]), 0)
        self.assertEqual(self.store.stats()["total"], 3)

    def test_import_skips_existing_memories(self):
        self.store.add("already here")
        self.assertEqual(self.store.import_items([{"text": "Already  Here"}, {"text": "new"}]), 1)

    def test_import_defaults(self):
        self.store.import_items([{"text": "bare minimum"}])
        item = self.store.export_all()[0]
        self.assertEqual((item["kind"], item["source"], item["pinned"], item["enabled"], item["uses"]),
                         ("fact", "import", False, True, 0))
        self.assertGreater(item["created"], 0)

    def test_import_keeps_valid_source_and_replaces_invalid(self):
        self.store.import_items([{"text": "one", "source": "auto"}, {"text": "two", "source": "alien"},
                                 {"text": "three", "source": None}, {"text": "four", "source": ["x"]}])
        sources = {i["text"]: i["source"] for i in self.store.export_all()}
        self.assertEqual(sources, {"one": "auto", "two": "import", "three": "import", "four": "import"})

    def test_import_skips_invalid_entries_without_failing(self):
        junk = [None, 5, "text", ["a"], {}, {"text": ""}, {"text": None}, {"text": 5}, {"text": "x" * 5000},
                {"text": "bad kind", "kind": "weird"}, {"text": "bad url", "url": "javascript:alert(1)"},
                {"text": "bad title", "title": 7}, {"text": "bad url type", "url": 5}, {"text": "fine"}]
        self.assertEqual(self.store.import_items(junk), 1)
        self.assertEqual(self.texts(self.store.export_all()), ["fine"])

    def test_import_rejects_non_lists_and_huge_batches(self):
        for bad in (None, {"text": "x"}, "text", 5):
            with self.assertRaises(ValueError):
                self.store.import_items(bad)
        with mock.patch.object(memory, "MAX_IMPORT", 3):
            with self.assertRaises(ValueError):
                self.store.import_items([{"text": str(i)} for i in range(4)])
            self.assertEqual(self.store.import_items([{"text": str(i)} for i in range(3)]), 3)

    def test_import_sanitises_numbers_and_flags(self):
        self.store.import_items([
            {"text": "future", "created": 4e9, "updated": 1.0, "uses": -3},
            {"text": "stringy", "created": "yesterday", "uses": "7", "pinned": "yes", "enabled": "no"},
            {"text": "flags", "pinned": 1, "enabled": 0, "uses": True},
            {"text": "good", "created": 1_700_000_000.5, "updated": 1_700_000_100.5, "uses": 4},
        ])
        items = {i["text"]: i for i in self.store.export_all()}
        self.assertLess(items["future"]["created"], 4e9)
        self.assertGreaterEqual(items["future"]["updated"], items["future"]["created"])
        self.assertEqual(items["future"]["uses"], 0)
        self.assertEqual((items["stringy"]["pinned"], items["stringy"]["enabled"], items["stringy"]["uses"]),
                         (False, True, 0))
        self.assertEqual((items["flags"]["pinned"], items["flags"]["enabled"], items["flags"]["uses"]),
                         (True, False, 0))
        self.assertEqual((items["good"]["created"], items["good"]["updated"], items["good"]["uses"]),
                         (1_700_000_000.5, 1_700_000_100.5, 4))

    def test_import_is_atomic_per_call(self):
        real_insert, calls = self.store._insert, []

        def fail_on_second(**fields):
            calls.append(fields["text"])
            if len(calls) == 2:
                raise RuntimeError("disk full")
            return real_insert(**fields)

        with mock.patch.object(self.store, "_insert", side_effect=fail_on_second):
            with self.assertRaises(RuntimeError):
                self.store.import_items([{"text": "one"}, {"text": "two"}])
        self.assertEqual(calls, ["one", "two"])
        self.assertEqual(self.store.stats()["total"], 0)


class DocumentTests(StoreCase):
    SENTENCES = [f"Sentence number {i} talks about topic {i % 7} in some detail." for i in range(120)]

    def doc(self):
        return "\n\n".join(" ".join(self.SENTENCES[i:i + 6]) for i in range(0, len(self.SENTENCES), 6))

    def chunks(self):
        return [r["text"] for r in self.db.query("SELECT text FROM memories WHERE kind = 'doc' ORDER BY id")]

    def test_chunks_are_bounded_aligned_and_complete(self):
        added = self.store.add_document("Guide", self.doc(), "https://example.com/guide", chunk_chars=300, overlap=100)
        chunks = self.chunks()
        self.assertEqual(added, len(chunks))
        self.assertGreater(added, 10)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 300)
            self.assertTrue(chunk.startswith("Sentence number"), chunk)
            self.assertTrue(chunk.endswith("detail."), chunk)
        joined = "\n".join(chunks)
        for sentence in self.SENTENCES:
            self.assertIn(sentence, joined)

    def test_stored_item_fields(self):
        self.store.add_document("  The   Guide ", "Some text worth keeping.", "https://example.com/g")
        item = self.store.list(kind="doc")["items"][0]
        self.assertEqual((item["text"], item["kind"], item["source"], item["title"], item["url"], item["pinned"]),
                         ("Some text worth keeping.", "doc", "user", "The Guide", "https://example.com/g", False))

    def test_overlap_repeats_whole_sentences(self):
        self.store.add_document("Guide", self.doc(), chunk_chars=300, overlap=100)
        chunks = self.chunks()
        for before, after in zip(chunks, chunks[1:]):
            first = after.split(". ")[0].split("\n\n")[0]
            self.assertIn(first, before)

    def test_zero_overlap_repeats_nothing(self):
        self.store.add_document("Guide", self.doc(), chunk_chars=300, overlap=0)
        chunks = self.chunks()
        seen = set()
        for chunk in chunks:
            for sentence in self.SENTENCES:
                if sentence in chunk:
                    self.assertNotIn(sentence, seen)
                    seen.add(sentence)
        self.assertEqual(len(seen), len(self.SENTENCES))

    def test_paragraph_boundaries_are_preferred(self):
        paragraphs = [f"Paragraph {i}. " + "alpha beta gamma. " * 10 for i in range(6)]
        self.store.add_document("P", "\n\n".join(paragraphs), chunk_chars=400, overlap=0)
        for chunk in self.chunks():
            self.assertTrue(chunk.startswith("Paragraph"), chunk)
            self.assertTrue(chunk.endswith("gamma."), chunk)

    def test_adding_the_same_document_twice_does_not_duplicate(self):
        first = self.store.add_document("Guide", self.doc(), chunk_chars=300, overlap=50)
        self.assertGreater(first, 1)
        self.assertEqual(self.store.add_document("Guide", self.doc(), chunk_chars=300, overlap=50), 0)
        self.assertEqual(self.store.add_document("Renamed", self.doc(), "https://example.com/x",
                                                 chunk_chars=300, overlap=50), 0)
        self.assertEqual(self.store.stats()["by_kind"]["doc"], first)

    def test_repeated_paragraphs_inside_one_document_collapse(self):
        paragraph = ("Same paragraph again and again. " * 4).strip()
        self.assertEqual(self.store.add_document("Echo", "\n\n".join([paragraph] * 5), chunk_chars=200, overlap=0), 1)

    def test_text_without_any_breaks_is_hard_split(self):
        blob = "".join(random.Random(7).choices(string.ascii_lowercase, k=5000))
        self.store.add_document("Blob", blob, chunk_chars=900, overlap=0)
        chunks = self.chunks()
        self.assertTrue(all(len(c) <= 900 for c in chunks))
        self.assertEqual("".join(chunks), blob)

    def test_long_sentence_is_split_at_spaces(self):
        words = [f"word{i}" for i in range(2000)]
        self.store.add_document("Words", " ".join(words), chunk_chars=500, overlap=0)
        chunks = self.chunks()
        self.assertTrue(all(len(c) <= 500 for c in chunks))
        self.assertEqual(" ".join(chunks).split(), words)

    def test_cjk_sentences_without_spaces(self):
        text = "".join(f"これは{i}番目の長い文章です。" for i in range(300))
        self.store.add_document("JP", text, chunk_chars=500, overlap=0)
        chunks = self.chunks()
        self.assertTrue(all(len(c) <= 500 for c in chunks))
        self.assertEqual("".join(chunks), text)

    def test_windows_line_endings_and_control_characters(self):
        text = "First paragraph.\r\n\r\nSecond\x00 paragraph.\r\n\r\n\r\n\r\nThird."
        self.store.add_document("Win", text, chunk_chars=200, overlap=0)
        self.assertEqual(self.chunks(), ["First paragraph.\n\nSecond paragraph.\n\nThird."])

    def test_chunk_limit(self):
        paragraphs = [f"P{i:04d} " + "x" * 140 for i in range(memory.MAX_CHUNKS)]
        self.assertEqual(self.store.add_document("Max", "\n\n".join(paragraphs), chunk_chars=200, overlap=0),
                         memory.MAX_CHUNKS)
        paragraphs = [f"Q{i:04d} " + "x" * 140 for i in range(memory.MAX_CHUNKS + 1)]
        with self.assertRaises(ValueError):
            self.store.add_document("Too many", "\n\n".join(paragraphs), chunk_chars=200, overlap=0)
        self.assertEqual(self.store.stats()["by_kind"]["doc"], memory.MAX_CHUNKS)

    def test_rejects_bad_input(self):
        bad_calls = [
            dict(text=""), dict(text="  \n\n "), dict(text="\x00\x01"), dict(text=None), dict(text=5),
            dict(text="x" * (memory.MAX_DOC_CHARS + 1)), dict(text="ok", source_url="javascript:alert(1)"),
            dict(text="ok", title=5), dict(text="ok", chunk_chars=50), dict(text="ok", chunk_chars=5000),
            dict(text="ok", chunk_chars="900"), dict(text="ok", chunk_chars=True), dict(text="ok", overlap=-1),
            dict(text="ok", overlap=451), dict(text="ok", overlap=1.5),
        ]
        for kwargs in bad_calls:
            with self.subTest(kwargs=str(kwargs)[:60]):
                with self.assertRaises(ValueError):
                    self.store.add_document(**{"title": "T", **kwargs})
        self.assertEqual(self.store.stats()["total"], 0)

    def test_default_parameters(self):
        text = "\n\n".join(f"Paragraph {i}. " + "Words and more words. " * 25 for i in range(12))
        self.assertGreater(self.store.add_document("Default", text), 5)
        self.assertTrue(all(len(c) <= 900 for c in self.chunks()))

    def test_dedupe_does_not_cross_kinds(self):
        self.store.add("Only paragraph here.", kind="note")
        self.assertEqual(self.store.add_document("Ünïcode ✓", "Only paragraph here."), 1)
        self.assertEqual(self.store.stats()["total"], 2)


class CandidateTests(StoreCase):
    def test_statements_that_are_found(self):
        table = [
            ("Remember that my cat is called Miso.", ["My cat is called Miso."]),
            ("remember: I use Neovim", ["I use Neovim."]),
            ("Please remember I'm allergic to peanuts!", ["I'm allergic to peanuts."]),
            ("Don't forget that the staging server is called box-7.", ["The staging server is called box-7."]),
            ("Keep in mind - answers should be short", ["Answers should be short."]),
            ("My name is Sam.", ["My name is Sam."]),
            ("Hi, my name is Sam and I live in Berlin.", ["My name is Sam and I live in Berlin."]),
            ("I prefer dark mode in every editor", ["I prefer dark mode in every editor."]),
            ("I hate long introductions.", ["I hate long introductions."]),
            ("I really like concise answers", ["I really like concise answers."]),
            ("I don't like emojis", ["I don't like emojis."]),
            ("I always use tabs", ["I always use tabs."]),
            ("I work on a Rust game engine.", ["I work on a Rust game engine."]),
            ("I live in Lisbon", ["I live in Lisbon."]),
            ("I use C#", ["I use C#."]),
            ("I\u2019m learning Rust", ["I'm learning Rust."]),
            ("I am building a home server", ["I am building a home server."]),
            ("Always answer in British English!", ["Always answer in British English."]),
            ("never use semicolons in my JavaScript", ["Never use semicolons in my JavaScript."]),
            ("Also, I speak Portuguese.", ["I speak Portuguese."]),
        ]
        for text, expected in table:
            with self.subTest(text=text):
                self.assertEqual(self.store.candidates_from_text(text), expected)

    def test_things_that_are_not_memories(self):
        for text in ["What's the capital of France?", "Do I like tea?", "I like tea?", "Remember to buy milk",
                     "Remember me?", "I like it", "I use this every day", "I like that answer", "I like your style",
                     "Write a function that remembers things", "Please summarize this article",
                     "Never mind, I found it", "Alice prefers tea", "She always uses tabs", "Always?",
                     "I remember that", "Do you remember that I like tea", "my", "Hello!", "", "   ", "\n\n"]:
            with self.subTest(text=text):
                self.assertEqual(self.store.candidates_from_text(text), [])

    def test_secrets_are_never_suggested(self):
        for text in ["My password is hunter2", "Remember that my API key is sk-abcdefghijklmnop12345678",
                     "remember: the token is ghp_abcdefghijklmnopqrstuvwxyz0123456789",
                     "I use AKIAABCDEFGHIJKLMNOP for deploys",
                     "Remember that the key is 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"]:
            with self.subTest(text=text):
                self.assertEqual(self.store.candidates_from_text(text), [])

    def test_ordinary_long_words_and_urls_are_not_secrets(self):
        out = self.store.candidates_from_text("Remember that docs live at https://example.com/a-very-long-path-name")
        self.assertEqual(len(out), 1)

    def test_multiple_lines_are_deduplicated_in_order(self):
        text = ("- I hate long intros.\n> never use semicolons\nJust a question: what is Rust?\n"
                "I HATE   long intros\nMy name is Sam. Remember that I use Linux.")
        self.assertEqual(self.store.candidates_from_text(text),
                         ["I hate long intros.", "Never use semicolons.", "My name is Sam.", "I use Linux."])

    def test_long_statements_are_cut_to_300_characters(self):
        out = self.store.candidates_from_text("Remember that " + "very " * 200 + "important")
        self.assertEqual(len(out), 1)
        self.assertLessEqual(len(out[0]), 300)
        self.assertTrue(out[0].endswith(memory.ELLIPSIS))
        short = self.store.candidates_from_text("I use " + "x" * 293)
        self.assertEqual(len(short[0]), 300)

    def test_result_is_capped(self):
        text = "\n".join(f"I like thing number {i}" for i in range(100))
        self.assertEqual(len(self.store.candidates_from_text(text)), memory.MAX_CANDIDATES)

    def test_odd_inputs(self):
        for bad in (None, 5, [], b"I like tea", {"a": 1}):
            self.assertEqual(self.store.candidates_from_text(bad), [])
        huge = self.store.candidates_from_text("I like tea " * 100_000)
        self.assertEqual(len(huge), 1)
        self.assertLessEqual(len(huge[0]), 300)

    def test_works_as_a_plain_function_call_too(self):
        self.assertEqual(MemoryStore.candidates_from_text("My name is Sam"), ["My name is Sam."])


def user(text):
    return {"role": "user", "content": text}


def assistant(text):
    return {"role": "assistant", "content": text}


class ExtractionMessageTests(unittest.TestCase):
    def test_shape_and_instructions(self):
        msgs = memory.extraction_messages([user("I use Arch btw"), assistant("Nice choice!")], 4)
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        system, body = msgs[0]["content"], msgs[1]["content"]
        self.assertIn("JSON array", system)
        self.assertIn("at most 4", system)
        for word in ("secrets", "passwords", "assistant"):
            self.assertIn(word, system)
        self.assertIn("User: I use Arch btw", body)
        self.assertIn("Assistant: Nice choice!", body)
        self.assertLess(body.index("User: I use Arch btw"), body.index("Assistant: Nice choice!"))

    def test_max_items_is_clamped(self):
        self.assertIn("at most 20", memory.extraction_messages([user("x")], 999)[0]["content"])
        self.assertIn("at most 1 strings", memory.extraction_messages([user("x")], 0)[0]["content"])
        self.assertIn("at most 5", memory.extraction_messages([user("x")], "junk")[0]["content"])

    def test_content_parts_and_other_roles(self):
        convo = [
            {"role": "system", "content": "SECRET SYSTEM PROMPT"},
            {"role": "user", "content": [{"type": "text", "text": "part one"}, {"type": "image_url", "image_url": {}},
                                         {"type": "text", "text": "part two"}]},
            {"role": "tool", "content": "TOOL OUTPUT"},
            {"role": "assistant", "content": None},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            "not a message", None, {"content": "no role"}, {"role": "user", "content": 5},
        ]
        body = memory.extraction_messages(convo, 3)[1]["content"]
        self.assertIn("User: part one\npart two", body)
        self.assertIn("Assistant: ok", body)
        for absent in ("SECRET SYSTEM PROMPT", "TOOL OUTPUT", "no role", "image_url"):
            self.assertNotIn(absent, body)

    def test_long_conversations_keep_the_newest_messages(self):
        convo = []
        for i in range(60):
            convo += [user(f"OLD-{i:02d} " + "q" * 400), assistant(f"reply {i} " + "a" * 400)]
        convo.append(user("NEWEST message about my dog Rex"))
        body = memory.extraction_messages(convo, 5)[1]["content"]
        self.assertIn("NEWEST message about my dog Rex", body)
        self.assertNotIn("OLD-00", body)
        self.assertIn("[earlier messages omitted]", body)
        self.assertLess(len(body), memory.TRANSCRIPT_CHARS + 400)

    def test_giant_messages_are_capped(self):
        convo = [user("U" * 100_000), assistant("A" * 100_000), user("final")]
        body = memory.extraction_messages(convo, 5)[1]["content"]
        self.assertLess(len(body), memory.TRANSCRIPT_CHARS + 400)
        self.assertIn("User: final", body)
        self.assertIn("A" * 100, body)
        self.assertNotIn("A" * (memory.ASSISTANT_MESSAGE_CHARS + 1), body)

    def test_empty_or_invalid_conversations(self):
        for convo in ([], None, "text", 5, [{"role": "system", "content": "x"}]):
            msgs = memory.extraction_messages(convo, 3)
            self.assertEqual([m["role"] for m in msgs], ["system", "user"])
            self.assertNotIn("User:", msgs[1]["content"])


class ParseExtractionTests(unittest.TestCase):
    def test_clean_json(self):
        self.assertEqual(memory.parse_extraction('["The user likes tea.", "The user uses Linux."]', 5),
                         ["The user likes tea.", "The user uses Linux."])
        self.assertEqual(memory.parse_extraction('  [ "a b" ,\n "c d" ]  ', 5), ["a b", "c d"])

    def test_code_fences_and_prose(self):
        reply = 'Sure! Here you go:\n```json\n["The user likes tea.", "The user lives in Oslo."]\n```\nHope that helps!'
        self.assertEqual(memory.parse_extraction(reply, 5), ["The user likes tea.", "The user lives in Oslo."])
        self.assertEqual(memory.parse_extraction('Result: ["one fact"] (done)', 5), ["one fact"])
        self.assertEqual(memory.parse_extraction('```\n["fenced without language"]\n```', 5),
                         ["fenced without language"])

    def test_empty_array_and_no_array(self):
        for reply in ("[]", "```json\n[]\n```", "Nothing worth remembering.", "", "   ", "{}", "[", "]", "[[", "null",
                      '{"a": "b"}', "[1, 2, 3]", "[null]", "[[], {}]"):
            self.assertEqual(memory.parse_extraction(reply, 5), [], repr(reply))

    def test_non_strings_are_dropped(self):
        self.assertEqual(memory.parse_extraction('[1, null, {"fact": "no"}, ["nested"], true, "kept fact", 2.5]', 5),
                         ["kept fact"])

    def test_leading_brackets_that_are_not_the_answer(self):
        self.assertEqual(memory.parse_extraction('As noted in [1], here: ["real fact"]', 5), ["real fact"])
        self.assertEqual(memory.parse_extraction("<think>draft [a, b]</think>\n[\"after thinking\"]", 5),
                         ["after thinking"])
        self.assertEqual(memory.parse_extraction('{"memories": ["inside an object"]}', 5), ["inside an object"])
        self.assertEqual(memory.parse_extraction('[["wrapped in another array"]]', 5), ["wrapped in another array"])

    def test_whitespace_dedupe_and_blank_items(self):
        reply = json.dumps(["  The user   likes\ttea. ", "the user likes tea.", "", "   ", "Another fact"])
        self.assertEqual(memory.parse_extraction(reply, 5), ["The user likes tea.", "Another fact"])

    def test_secrets_are_dropped(self):
        reply = json.dumps([
            "The user's OpenAI key is sk-proj-abcdefghijklmnopqrstuvwx",
            "The password is hunter2",
            "Their GitHub token: ghp_abcdefghijklmnopqrstuvwxyz0123456789",
            "AWS key AKIAIOSFODNN7EXAMPLE is used for backups",
            "api_key = abc123",
            "Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456",
            "Deploy hash 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
            "The user prefers dark mode.",
            "The user keeps passwords in a password manager.",
            "The docs are at https://example.com/a-very-long-descriptive-path-name-for-the-page",
        ])
        self.assertEqual(memory.parse_extraction(reply, 20), [
            "The user prefers dark mode.", "The user keeps passwords in a password manager.",
            "The docs are at https://example.com/a-very-long-descriptive-path-name-for-the-page"])

    def test_long_items_are_truncated_to_300(self):
        out = memory.parse_extraction(json.dumps(["x " * 400, "y" * 300]), 5)
        self.assertEqual(len(out[0]), 300)
        self.assertTrue(out[0].endswith(memory.ELLIPSIS))
        self.assertEqual(out[1], "y" * 300)

    def test_cap_on_item_count(self):
        reply = json.dumps([f"fact number {i}" for i in range(30)])
        self.assertEqual(len(memory.parse_extraction(reply, 5)), 5)
        self.assertEqual(memory.parse_extraction(reply, 5)[0], "fact number 0")
        self.assertEqual(memory.parse_extraction(reply, 0), [])
        self.assertEqual(memory.parse_extraction(reply, -3), [])
        self.assertEqual(len(memory.parse_extraction(reply, 3.9)), 3)

    def test_output_cut_off_by_the_token_limit(self):
        self.assertEqual(memory.parse_extraction('["The user likes tea.", "The user uses Linux.", "The user wor', 5),
                         ["The user likes tea.", "The user uses Linux."])
        self.assertEqual(memory.parse_extraction('Here: ["first \\"quoted\\" fact", "second', 5),
                         ['first "quoted" fact'])
        self.assertEqual(memory.parse_extraction('["unfinished', 5), [])

    def test_unicode_and_escapes(self):
        self.assertEqual(memory.parse_extraction('["Benutzer wohnt in Köln", "\\u00fcber alles", "line\\nbreak"]', 5),
                         ["Benutzer wohnt in Köln", "über alles", "line break"])

    def test_garbage_never_raises(self):
        for reply in (None, 5, [], b'["x"]', "[" * 5000, "[" * 100_000, '["' * 5000, "]" * 100, "\x00[\x00",
                      '["a",,"b"]', "[\"a\" \"b\"]", "[" + '"a",' * 100_000):
            with self.subTest(reply=repr(reply)[:30]):
                self.assertIsInstance(memory.parse_extraction(reply, 5), list)

    def test_large_replies_are_bounded(self):
        reply = "noise " * 10_000 + '["too late"]'
        self.assertEqual(memory.parse_extraction(reply, 5), [])


class ConcurrencyTests(StoreCase):
    def run_threads(self, target, n=8):
        errors = []

        def wrapper(i):
            try:
                target(i)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=wrapper, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        self.assertEqual(errors, [])

    def test_concurrent_duplicate_adds_leave_one_row(self):
        ids = set()

        def work(_thread):
            for _round in range(5):
                ids.add(self.store.add("  The   SAME memory ")["id"])

        self.run_threads(work)
        self.assertEqual(len(ids), 1)
        self.assertEqual(self.store.stats()["total"], 1)

    def test_concurrent_distinct_adds_and_searches(self):
        def work(i):
            for j in range(10):
                self.store.add(f"thread {i} remembers item {j}")
                self.store.search("remembers item")

        self.run_threads(work)
        self.assertEqual(self.store.stats()["total"], 80)


class ModuleTests(unittest.TestCase):
    def test_query_tokens(self):
        self.assertEqual(memory._query_tokens("What is the Rust-based game engine?"),
                         ["rust", "based", "game", "engine"])
        self.assertEqual(memory._query_tokens("tea tea TEA"), ["tea"])
        self.assertEqual(memory._query_tokens("a b c OR * \" 🙂"), [])
        self.assertEqual(len(memory._query_tokens(" ".join(f"w{i}" for i in range(100)))), memory.MAX_QUERY_TOKENS)
        self.assertEqual(memory._query_tokens(None), [])

    def test_stem(self):
        self.assertEqual([memory._stem(w) for w in ("bikes", "parties", "glass", "bus", "this", "is", "cats", "go")],
                         ["bike", "party", "glass", "bus", "this", "is", "cat", "go"])


if __name__ == "__main__":
    unittest.main()
