"""Unicode normalization and cleaning for Urdu text.

Diacritic policy
----------------
Urdu is written in the Arabic script with an extended Perso-Arabic character
set. We apply NFC (canonical composition) so that visually identical strings
have one code-point sequence. We do NOT strip harakat (short-vowel diacritics
such as zabar/zer/pesh): they are rare in running Urdu text but meaningful when
present, and removing them would corrupt the small share of vocalised text
(religious quotations, dictionaries). We DO normalise a handful of characters
that are genuine encoding variants rather than distinct letters:

  - Arabic Kaf U+0643 -> Urdu Keheh U+06A9
  - Arabic Yeh  U+064A -> Farsi Yeh   U+06CC
  - Arabic Alef Maksura U+0649 -> Farsi Yeh U+06CC
  - Zero-width non-joiner is kept (it carries orthographic meaning in Urdu),
    but zero-width space / joiner / BOM and other invisibles are removed.

This mirrors the normalisation used by most published Urdu NLP corpora and
keeps the tokenizer vocabulary from splitting on encoding accidents.
"""

import re
import unicodedata

# Character folding map: encoding variants -> canonical Urdu form.
_FOLD = {
    "ك": "ک",  # Arabic kaf -> Urdu keheh
    "ي": "ی",  # Arabic yeh -> Farsi yeh
    "ى": "ی",  # Alef maksura -> Farsi yeh
    "ۀ": "ہ",  # heh with hamza above -> heh goal (common variant)
    "​": "",        # zero-width space
    "‍": "",        # zero-width joiner
    "﻿": "",        # BOM / zero-width no-break space
    " ": " ",       # non-breaking space -> space
    "ـ": "",        # tatweel / kashida (decorative elongation)
}
_FOLD_RE = re.compile("|".join(re.escape(k) for k in _FOLD if k))

# Arabic-script block plus common punctuation/digits used in Urdu.
_ARABIC_RANGES = (
    (0x0600, 0x06FF),   # Arabic
    (0x0750, 0x077F),   # Arabic Supplement
    (0x08A0, 0x08FF),   # Arabic Extended-A
    (0xFB50, 0xFDFF),   # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),   # Arabic Presentation Forms-B
)

_WS_RE = re.compile(r"[ \t ]+")
_NEWLINES_RE = re.compile(r"\n{3,}")


def is_arabic_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _ARABIC_RANGES)


def arabic_ratio(text: str) -> float:
    """Share of letters that are in the Arabic script block."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    ara = sum(1 for c in letters if is_arabic_char(c))
    return ara / len(letters)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _FOLD_RE.sub(lambda m: _FOLD[m.group(0)], text)
    text = _WS_RE.sub(" ", text)
    text = _NEWLINES_RE.sub("\n\n", text)
    return text.strip()


# Language-ID choice: we use a script-range heuristic (arabic_ratio) rather than
# a fastText model. Justification: Urdu shares its script only with other
# Perso-Arabic languages, none of which appear at meaningful volume in these
# sources (a Wikipedia dump filtered to `ur`, and Leipzig corpora already tagged
# `urd`). The dominant contaminant is embedded English/Latin, which the ratio
# catches directly. A heuristic keeps the pipeline dependency-light and fully
# reproducible on CPU without downloading a 126MB lid model.
MIN_ARABIC_RATIO = 0.65
MIN_CHARS = 40
MIN_WORDS = 6


def keep_line(text: str) -> bool:
    """Boilerplate / language filter for a single normalized paragraph."""
    if len(text) < MIN_CHARS:
        return False
    if len(text.split()) < MIN_WORDS:
        return False
    if arabic_ratio(text) < MIN_ARABIC_RATIO:
        return False
    # drop lines that are mostly digits/punctuation (tables, infobox leftovers)
    alpha = sum(1 for c in text if c.isalpha())
    if alpha / max(len(text), 1) < 0.5:
        return False
    return True
