"""Source readers. Each yields raw document dicts {text, source, id/title}.

Three sources are wired up, all downloadable without authentication:

  * wiki    : a MediaWiki XML dump (bz2) of Urdu Wikipedia. Wikitext is stripped
              with mwparserfromhell.
  * leipzig : Wortschatz Leipzig sentence corpora (tar.gz), one sentence per
              tab-separated line in `*-sentences.txt`.
  * plain   : a plain-text corpus file, optionally .xz or .gz compressed, in the
              CC-100 layout: consecutive non-empty lines form one document and a
              blank line ends it.
"""

import bz2
import gzip
import lzma
import re
import tarfile
import xml.etree.ElementTree as ET

import mwparserfromhell

# Redirect / disambiguation pages and namespaces we skip.
_REDIRECT_RE = re.compile(r"^\s*#(REDIRECT|تحویل|رجوع مکرر)", re.IGNORECASE)


def read_wiki_dump(path, max_docs=None):
    """Stream articles from a MediaWiki pages-articles bz2 dump."""
    n = 0
    with bz2.open(path, "rb") as fh:
        title, text, ns = None, None, None
        for event, elem in ET.iterparse(fh, events=("end",)):
            tag = elem.tag.split("}")[-1]
            if tag == "title":
                title = elem.text
            elif tag == "ns":
                ns = elem.text
            elif tag == "text":
                text = elem.text
            elif tag == "page":
                if ns == "0" and text and title and not _REDIRECT_RE.match(text):
                    body = _strip_wikitext(text)
                    if body:
                        yield {"text": body, "source": "wikipedia", "title": title}
                        n += 1
                        if max_docs and n >= max_docs:
                            elem.clear()
                            return
                title, text, ns = None, None, None
                elem.clear()


def _strip_wikitext(wikitext):
    try:
        code = mwparserfromhell.parse(wikitext)
        text = code.strip_code(normalize=True, collapse=True)
    except Exception:
        return ""
    # remove leftover reference/file markup lines
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(("|", "!", "{", "}")):
            continue
        lines.append(line)
    return "\n".join(lines)


def read_plain_text(path, corpus_name, max_docs=None):
    """Stream documents from a plain-text file (CC-100 layout, see module doc)."""
    p = str(path)
    if p.endswith(".xz"):
        fh = lzma.open(p, "rt", encoding="utf-8", errors="ignore")
    elif p.endswith(".gz"):
        fh = gzip.open(p, "rt", encoding="utf-8", errors="ignore")
    else:
        fh = open(p, "r", encoding="utf-8", errors="ignore")
    n = 0
    doc_lines = []
    with fh:
        for line in fh:
            line = line.strip()
            if line:
                doc_lines.append(line)
                continue
            if doc_lines:
                yield {"text": "\n".join(doc_lines), "source": corpus_name}
                doc_lines = []
                n += 1
                if max_docs and n >= max_docs:
                    return
        if doc_lines:
            yield {"text": "\n".join(doc_lines), "source": corpus_name}


# Declarative Urdu sentence templates per Wikidata relation (see
# pipeline/fetch_wikidata.py). Two phrasings each for mild lexical variety;
# alternated deterministically by row index, not randomly, so a rerun is
# reproducible.
_WIKIDATA_TEMPLATES = {
    "capital": [
        "{subject} کا دارالحکومت {object} ہے۔",
        "{object} {subject} کا دارالحکومت ہے۔",
    ],
    "official_language": [
        "{subject} کی سرکاری زبان {object} ہے۔",
        "{object} {subject} کی سرکاری زبان ہے۔",
    ],
    "currency": [
        "{subject} کی کرنسی {object} ہے۔",
        "{object} {subject} کی کرنسی ہے۔",
    ],
    "continent": [
        "{subject} {object} کا ملک ہے۔",
        "{subject} براعظم {object} میں واقع ہے۔",
    ],
}


def read_wikidata_facts(path, max_docs=None):
    """Render Wikidata subject/relation/object triples into declarative
    Urdu sentences. `path` is the JSONL written by pipeline.fetch_wikidata."""
    import json

    n = 0
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            templates = _WIKIDATA_TEMPLATES.get(rec["relation"])
            if not templates:
                continue
            template = templates[i % len(templates)]
            text = template.format(subject=rec["subject"], object=rec["object"])
            yield {"text": text, "source": "wikidata"}
            n += 1
            if max_docs and n >= max_docs:
                return


def read_leipzig_tar(path, corpus_name, max_docs=None):
    """Stream sentences from a Leipzig corpus tar.gz (uses *-sentences.txt)."""
    n = 0
    with tarfile.open(path, "r:gz") as tar:
        member = None
        for m in tar.getmembers():
            if m.name.endswith("-sentences.txt"):
                member = m
                break
        if member is None:
            return
        fh = tar.extractfile(member)
        for raw in fh:
            line = raw.decode("utf-8", errors="ignore").rstrip("\n")
            if not line:
                continue
            # format: "<id>\t<sentence>"
            parts = line.split("\t", 1)
            sentence = parts[1] if len(parts) == 2 else parts[0]
            yield {"text": sentence, "source": corpus_name}
            n += 1
            if max_docs and n >= max_docs:
                return
