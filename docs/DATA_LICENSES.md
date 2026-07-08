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

## 3. CC-100 (Urdu), used from the v2 corpus (2026-07-08) onward

- URL: https://data.statmt.org/cc-100/ur.txt.xz
- Size: ~884 MB compressed (xz).
- License: "No claims of intellectual property are made on the work of
  preparation of the corpus" by CC-100's preparers; users are bound by the
  Common Crawl Foundation's terms of use
  (https://commoncrawl.org/terms-of-use), since the corpus is a filtered
  extract of Common Crawl. Not a standard open-content license (no CC BY/SA
  label), but openly and freely redistributable for research use.
- Use here: plain-text, blank-line-delimited documents, read directly by
  `pipeline.sources.read_plain_text` (handles `.xz`/`.gz`/plain), normalized
  and filtered the same as the other sources.
- Not used in the v1 (2026-07-06) corpus, skipped then to keep the download
  modest and the first proof run fast.

## 4. Wikidata (fact triples), added 2026-07-08 for the planned `medium` run

- Endpoint: https://query.wikidata.org/sparql (public SPARQL Query Service,
  no authentication).
- License: Wikidata's own content is dedicated to the public domain under
  CC0 1.0 (https://www.wikidata.org/wiki/Wikidata:Licensing). No restriction
  inherited from this source, unlike Leipzig's CC BY-NC.
- Use here: `pipeline.fetch_wikidata` queries four relations (a country's
  capital, official language, currency, continent) restricted to rows where
  both the subject and object have a real Urdu label (rows falling back to
  another language are dropped), and writes the raw subject/relation/object
  triples to a JSONL file. `pipeline.sources.read_wikidata_facts` renders
  each triple into one of two fixed Urdu declarative-sentence templates per
  relation (alternated by row index for lexical variety) and feeds it into
  the pipeline as a `wikidata` source. Real test run on 2026-07-08 returned
  890 triples across all four relations; 865 survived the corpus-wide
  MinHash near-dedup (2.81% dropped, mostly countries that legitimately
  share an object, e.g. several sharing English as an official language).
- Purpose: distinct entities from the eval set's own 45 cloze items, chosen
  because Wikidata's fact-storage-capacity research (see README) motivates
  giving the model dense, repeated, unambiguous exposure to this specific
  fact category rather than diffuse prose mentions of the same fact.

## Sources considered and skipped

- OSCAR / mC4 Urdu: gated behind Hugging Face auth or a click-through, so
  excluded per the no-auth rule.

## Regeneration

Raw downloads are kept out of the repository (they live in a scratch directory
during the build and are deleted afterward). To rebuild the v2 corpus, download
the three URLs in sections 1-3 into a directory and run the command in the
top-level README under "Reproduce the corpus"; drop the CC-100 file and the
upsample flags to rebuild the smaller v1 corpus instead. To add Wikidata
facts for the `medium` run, also run `pipeline.fetch_wikidata` into the same
directory first, then pass `--wikidata wikidata_facts.jsonl` to
`pipeline.build_corpus`.
