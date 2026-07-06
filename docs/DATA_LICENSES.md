# Data sources and licenses

All sources below are downloadable over plain HTTPS without authentication. Each
was verified reachable from this machine on 2026-07-06.

## 1. Urdu Wikipedia dump

- URL: https://dumps.wikimedia.org/urwiki/latest/urwiki-latest-pages-articles.xml.bz2
- Size: ~455 MB compressed (bz2).
- License: text is dual-licensed under Creative Commons Attribution-ShareAlike
  4.0 (CC BY-SA 4.0) and the GNU Free Documentation License (GFDL). See
  https://dumps.wikimedia.org/legal.html and
  https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use .
- Attribution: "Urdu Wikipedia contributors". Derivatives of this text must be
  shared under CC BY-SA 4.0.
- Use here: article wikitext (namespace 0 only) is parsed with
  mwparserfromhell, split into paragraphs, normalized, and filtered.

## 2. Wortschatz Leipzig Corpora (Urdu)

- Project: Leipzig Corpora Collection, University of Leipzig.
- Files used:
  - urd_newscrawl_2016_1M  (1,000,000 sentences, news crawl 2016)
    https://downloads.wortschatz-leipzig.de/corpora/urd_newscrawl_2016_1M.tar.gz
  - urd_newscrawl_2011_300K (300,000 sentences, news crawl 2011)
    https://downloads.wortschatz-leipzig.de/corpora/urd_newscrawl_2011_300K.tar.gz
- License: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0), stated
  in each archive's terms and on
  https://wortschatz.uni-leipzig.de/en/download . The corpora are distributed
  for research and non-commercial use. Attribution: D. Goldhahn, T. Eckart, U.
  Quasthoff, "Building Large Monolingual Dictionaries at the Leipzig Corpora
  Collection", LREC 2012.
- Note on the NC clause: because Leipzig is CC BY-NC, any model or artifact that
  reuses this text inherits a non-commercial restriction. This is documented so
  a downstream user knows the corpus is for research/non-commercial use. If a
  fully permissive corpus is required, retrain the tokenizer and model on the
  Wikipedia subset alone (the pipeline supports dropping sources).

## Sources considered and skipped

- CC-100 (ur) from data.statmt.org: openly downloadable, CC-BY-like terms
  derived from CommonCrawl. Skipped for the phase-1 build only to keep the
  download modest and the proof run fast; the pipeline can ingest it by pointing
  `read_leipzig_tar`-style readers at a plain-text source. Not used here, so not
  claimed in the corpus numbers.
- OSCAR / mC4 Urdu: gated behind Hugging Face auth or a click-through, so
  excluded per the no-auth rule.

## Regeneration

Raw downloads are kept out of the repository (they live in a scratch directory
during the build and are deleted afterward). To rebuild the corpus, download the
three URLs above into a directory and run the command in the top-level README
under "Reproduce the corpus".
