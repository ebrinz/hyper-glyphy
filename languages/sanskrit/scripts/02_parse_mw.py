"""
Monier-Williams Parser: extract Sanskrit-English glosses from the Cologne
CDSL MW XML (1899 edition, 2020 digitization).

Source: https://www.sanskrit-lexicon.uni-koeln.de/scans/MWScan/2020/downloads/mwxml.zip
  -> xml/mw.xml (67 MB). Entry elements are <H1>..<H4> plus suffixed
  variants (<H1A>, <H2B>, <H1E>, ...). Each has <h><key1>SLP1 headword</key1>
  and a <body> mixing English gloss text with tagged non-English material:
  <s> (SLP1 Sanskrit), <ls> (literary sources), <ab> (abbreviations),
  <lex> (grammar), <gk>/<lang>/<etym> (etymology), <ns> (proper names), etc.

Gloss extraction: collect body text OUTSIDE excluded tags, strip
parentheticals, split on [;,], trim noise segments. First gloss segment is
`gloss_first` (06 falls back to later segments — including when a segment is
negation-led, e.g. "not injuring anything" -> falls through to "harmlessness").

SLP1 -> IAST at parse time via a deterministic in-repo table (no dependency).

Output: data/dictionaries/mw_glosses.json
  [{key1, lemma_iast, lemma_norm, glosses: [...], gloss_first}]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from languages.sanskrit.scripts.sanskrit_normalize import normalize_sanskrit_token  # noqa: E402

DATA_DICTS = Path(__file__).parent.parent / "data" / "dictionaries"
MW_XML_DEFAULT = Path(__file__).parent.parent / "data" / "raw" / "mw" / "xml" / "mw.xml"

# Full SLP1 set including digraph expansions (spec 2026-07-13).
SLP1_TO_IAST = {
    "a": "a", "A": "ā", "i": "i", "I": "ī", "u": "u", "U": "ū",
    "f": "ṛ", "F": "ṝ", "x": "ḷ", "X": "ḹ",
    "e": "e", "E": "ai", "o": "o", "O": "au",
    "M": "ṃ", "H": "ḥ", "~": "m̐",
    "k": "k", "K": "kh", "g": "g", "G": "gh", "N": "ṅ",
    "c": "c", "C": "ch", "j": "j", "J": "jh", "Y": "ñ",
    "w": "ṭ", "W": "ṭh", "q": "ḍ", "Q": "ḍh", "R": "ṇ",
    "t": "t", "T": "th", "d": "d", "D": "dh", "n": "n",
    "p": "p", "P": "ph", "b": "b", "B": "bh", "m": "m",
    "y": "y", "r": "r", "l": "l", "v": "v",
    "S": "ś", "z": "ṣ", "s": "s", "h": "h", "L": "ḻ",
}

_ENTRY_TAG_RE = re.compile(r"^H[1-4][A-E]?$")
_PAREN_RE = re.compile(r"\([^)]*\)")
_NOISE_SEGMENTS = {"&c", "etc", "ib", "q.v", "cf"}

# Body child tags whose TEXT is non-English (their tails are kept — English
# gloss text frequently sits in tails, e.g. "(<ab>fr.</ab> <s>div</s>) heavenly").
EXCLUDE_TAGS = {
    "s", "s1", "ls", "ab", "lex", "hom", "info", "lang", "gk", "etym",
    "ns", "bio", "bot", "root", "pb", "srs", "shc", "vlex", "abE", "pcol",
}


def slp1_to_iast(slp: str) -> str:
    """Deterministic char-by-char SLP1 -> IAST; unknown chars pass through."""
    return "".join(SLP1_TO_IAST.get(ch, ch) for ch in slp)


def _localname(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _collect_english_text(el) -> str:
    """Text of `el` excluding the text of EXCLUDE_TAGS subtrees (tails kept)."""
    parts = [el.text or ""]
    for child in el:
        if _localname(child.tag) not in EXCLUDE_TAGS:
            parts.append(_collect_english_text(child))
        parts.append(child.tail or "")
    return "".join(parts)


def extract_glosses(body_el) -> list[str]:
    """English gloss segments from a <body>: strip parentheticals, split on
    [;,], drop noise. Downstream 06 handles negation-led segments."""
    text = _collect_english_text(body_el)
    text = _PAREN_RE.sub(" ", text)
    glosses: list[str] = []
    for seg in re.split(r"[;,]", text):
        seg = re.sub(r"\s+", " ", seg).strip(" .:—–-'’\"")
        if not seg or seg.lower() in _NOISE_SEGMENTS:
            continue
        if not re.search(r"[A-Za-z]{2}", seg):
            continue
        glosses.append(seg)
    return glosses


def parse_mw_file(path: Path) -> list[dict]:
    """Parse mw.xml. Returns deduplicated entries (first occurrence per
    lemma_norm wins — usually the main entry, matching the LSJ recipe)."""
    entries: list[dict] = []
    for _, elem in ET.iterparse(str(path), events=("end",)):
        tag = _localname(elem.tag)
        if not _ENTRY_TAG_RE.match(tag):
            continue
        h = elem.find("h")
        body = elem.find("body")
        key1 = h.findtext("key1") if h is not None else None
        if not key1 or body is None:
            elem.clear()
            continue
        glosses = extract_glosses(body)
        if glosses:
            lemma_iast = slp1_to_iast(key1)
            lemma_norm = normalize_sanskrit_token(lemma_iast)
            if lemma_norm:
                entries.append({
                    "key1": key1,
                    "lemma_iast": lemma_iast,
                    "lemma_norm": lemma_norm,
                    "glosses": glosses,
                    "gloss_first": glosses[0],
                })
        elem.clear()

    seen: dict[str, dict] = {}
    for entry in entries:
        seen.setdefault(entry["lemma_norm"], entry)
    return list(seen.values())


def main(mw_xml: Path = MW_XML_DEFAULT):
    if not mw_xml.exists():
        raise SystemExit(
            f"MW XML not found at {mw_xml}. Run the documented fetch step "
            "(languages/sanskrit/README.md): download mwxml.zip from Cologne "
            "CDSL and unzip into data/raw/mw/."
        )
    DATA_DICTS.mkdir(parents=True, exist_ok=True)

    deduped = parse_mw_file(mw_xml)
    output_path = DATA_DICTS / "mw_glosses.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)

    print(f"Entries after dedup by lemma_norm: {len(deduped)}")
    print(f"Saved to: {output_path}")
    print("\nFirst 10 glosses:")
    for e in deduped[:10]:
        print(f"  {e['lemma_iast']:>20s}  ->  {e['gloss_first']!r}")


if __name__ == "__main__":
    main()
