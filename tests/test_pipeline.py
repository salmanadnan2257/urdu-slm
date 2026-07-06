"""Tests for normalization, filtering, and dedup logic."""

from pipeline.normalize import normalize, keep_line, arabic_ratio
from pipeline.dedup import Deduper, exact_key


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
