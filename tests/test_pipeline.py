"""Tests for normalization, filtering, and dedup logic."""

import json
import tempfile
from pathlib import Path

from pipeline.normalize import normalize, keep_line, arabic_ratio
from pipeline.dedup import Deduper, exact_key
from pipeline.sources import read_wikidata_facts


def test_normalize_folds_arabic_variants():
    # Arabic kaf + yeh should fold to Urdu keheh + farsi yeh
    assert normalize("كيا") == "کیا"


def test_normalize_strips_invisibles_and_ws():
    assert normalize("سلام‌ـ  دنیا") == "سلام‌ دنیا"  # ZWNJ kept, tatweel gone
    assert normalize("a\t\tb") == "a b"


def test_arabic_ratio():
    assert arabic_ratio("یہ اردو ہے") > 0.9
    assert arabic_ratio("this is english") == 0.0
    assert 0.2 < arabic_ratio("اردو and english") < 0.4


def test_keep_line_filters():
    assert not keep_line("مختصر")                       # too short
    assert not keep_line("this is a long english sentence with many words here")
    good = "پاکستان جنوبی ایشیا کا ایک اہم ملک ہے جہاں مختلف زبانیں بولی جاتی ہیں"
    assert keep_line(good)


def test_exact_dedup():
    d = Deduper()
    assert d.keep("یہ ایک جملہ ہے جو دہرایا جائے گا")
    assert not d.keep("یہ ایک جملہ ہے جو دہرایا جائے گا")  # exact dup
    assert d.n_exact == 1


def test_near_dedup():
    d = Deduper(threshold=0.6)
    base = "پاکستان ایک ملک ہے جہاں اردو بولی جاتی ہے اور یہ جنوبی ایشیا میں واقع ہے"
    near = "پاکستان ایک ملک ہے جہاں اردو بولی جاتی ہے اور یہ جنوبی ایشیا میں واقع ہے۔"
    assert d.keep(base)
    assert not d.keep(near)   # near duplicate caught by MinHash
    assert d.n_near == 1


def test_exact_key_stable():
    assert exact_key("test") == exact_key("test")


def test_read_wikidata_facts_renders_declarative_sentences():
    rows = [
        {"relation": "capital", "subject": "جاپان", "object": "ٹوکیو"},
        {"relation": "official_language", "subject": "کینیڈا", "object": "انگریزی"},
        {"relation": "unknown_relation", "subject": "x", "object": "y"},
    ]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "wikidata_facts.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        docs = list(read_wikidata_facts(path))
    # the unknown relation has no template and is skipped
    assert len(docs) == 2
    assert all(d["source"] == "wikidata" for d in docs)
    assert "جاپان" in docs[0]["text"] and "ٹوکیو" in docs[0]["text"]
    assert "کینیڈا" in docs[1]["text"] and "انگریزی" in docs[1]["text"]
