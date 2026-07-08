"""Fetch declarative-fact triples from Wikidata's public SPARQL endpoint.

Standalone script, run before pipeline.build_corpus, same pattern as the
Wikipedia/Leipzig downloads: writes a raw file into --raw-dir for the pipeline
to read later. No authentication; the endpoint is public and rate-limited by
courtesy (one query at a time, a short pause between queries).

Relations chosen are ones with a single clean Urdu-label object per subject,
so each triple renders into one unambiguous declarative sentence: a country's
capital, official language, currency, and continent. This is a deliberately
different fact category from Urdu Wikipedia's own text (which already covers
these entities but not necessarily in this dense one-fact-per-sentence form),
meant to give training more repeated, clean exposure to exactly the kind of
fact the eval/cloze.jsonl harness probes, using different entities than the
eval set's own 45 items.

Run:
  python -m pipeline.fetch_wikidata --raw-dir <scratch/raw> --out wikidata_facts.jsonl
"""

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "urdu-slm-research/1.0 (educational corpus-building project)"

# relation name -> (SPARQL property id, SPARQL query template)
QUERIES = {
    "capital": """
        SELECT ?subjectLabel ?objectLabel WHERE {
          ?subject wdt:P36 ?object.
          ?subject wdt:P31 wd:Q6256.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "ur". }
        }
    """,
    "official_language": """
        SELECT ?subjectLabel ?objectLabel WHERE {
          ?subject wdt:P37 ?object.
          ?subject wdt:P31 wd:Q6256.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "ur". }
        }
    """,
    "currency": """
        SELECT ?subjectLabel ?objectLabel WHERE {
          ?subject wdt:P38 ?object.
          ?subject wdt:P31 wd:Q6256.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "ur". }
        }
    """,
    "continent": """
        SELECT ?subjectLabel ?objectLabel WHERE {
          ?subject wdt:P30 ?object.
          ?subject wdt:P31 wd:Q6256.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "ur". }
        }
    """,
}


def run_query(sparql: str) -> list[dict]:
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": sparql, "format": "json"})
    req = urllib.request.Request(url, headers={
        "Accept": "application/sparql-results+json",
        "User-Agent": USER_AGENT,
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    out = []
    for row in data["results"]["bindings"]:
        subj = row.get("subjectLabel", {}).get("value")
        obj = row.get("objectLabel", {}).get("value")
        # Skip rows where the label service fell back to a non-Urdu language
        # (Wikidata's label service falls back to English/the item's own
        # language when no Urdu label exists; we only want real Urdu text).
        if row.get("subjectLabel", {}).get("xml:lang") != "ur":
            continue
        if row.get("objectLabel", {}).get("xml:lang") != "ur":
            continue
        if subj and obj:
            out.append({"subject": subj, "object": obj})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out", default="wikidata_facts.jsonl")
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between queries, be a courteous client")
    args = ap.parse_args()

    out_path = Path(args.raw_dir) / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for relation, sparql in QUERIES.items():
            rows = run_query(sparql)
            for row in rows:
                rec = {"relation": relation, "subject": row["subject"], "object": row["object"]}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1
            print(f"[wikidata] {relation}: {len(rows)} Urdu-labeled triples")
            time.sleep(args.sleep)

    print(f"[wikidata] wrote {total} triples -> {out_path}")


if __name__ == "__main__":
    main()
