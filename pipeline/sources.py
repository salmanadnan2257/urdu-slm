"""Source readers. Each yields raw document dicts {text, source, id/title}.

Two sources are wired up, both downloadable without authentication:

  * wiki    : a MediaWiki XML dump (bz2) of Urdu Wikipedia. Wikitext is stripped
              with mwparserfromhell.
  * leipzig : Wortschatz Leipzig sentence corpora (tar.gz), one sentence per
              tab-separated line in `*-sentences.txt`.
"""

import bz2
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
