"""
TLHdig XML Parser: convert HPM XML to hyper-glyphy lemma + text JSON.

Source: TLHdig 0.2.0-beta (Zenodo 15459134), 22k XML files organized by
CTH (Catalogue des Textes Hittites) number.

Each <w> element carries:
  - trans: transliterated form (e.g., "nuššan", "atanzi", "DINGIR-LIM")
  - mrp0sel: selected analysis index ("1", "2", ..., or "DEL", or " " (default = mrp1))
  - mrp1..mrp5: morphological analyses in format LEMMA@GLOSS@FEATURES@CATEGORY@
  - Children: <sGr>Sumerogram</sGr>, <aGr>Akkadogram</aGr>, <d>determinative</d>

Output:
  - data/raw/hittite_texts.json:   [{p_number, lines: [str], source: "TLHdig"}]
  - data/raw/hittite_lemmas.json:  [{form, cf, gw (German!), pos, lang, ...}]
  - data/raw/hittite_heterograms.json: aggregate {sumerogram: count, akkadogram: count}

Glosses are GERMAN (Gemma is multilingual; downstream encoding handles translation).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from tqdm import tqdm

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"

# Default TLHdig extract location; override via env if needed.
TLHDIG_ROOT_DEFAULT = Path("/tmp/tlhdig/TLHbasisONLINE25.1_ZENODO")


_MRP_SEL_NUMERIC = re.compile(r"(\d+)")


def parse_mrp(mrp_str: str | None) -> dict | None:
    """Parse one mrp{N} attribute value: LEMMA@GLOSS@FEATURES@CATEGORY@(extra).

    Empty/None input returns None.
    Fields are split on '@'; trailing empty segments are tolerated.
    """
    if not mrp_str:
        return None
    parts = mrp_str.split("@")
    # Expected at least: LEMMA@GLOSS@FEATURES@CATEGORY@
    while len(parts) < 5:
        parts.append("")
    return {
        "lemma": parts[0].strip(),
        "gloss": parts[1].strip(),
        "features": parts[2].strip(),
        "category": parts[3].strip(),
        "extra": parts[4].strip() if len(parts) >= 5 else "",
    }


def select_mrp(attrs: dict[str, str]) -> dict | None:
    """Pick the canonical mrp{N} based on mrp0sel.

    mrp0sel values:
      - "DEL" or similar → word is deleted/uncertain; return None
      - " 1 " / "1" → use mrp1
      - " N " → use mrpN
      - " " (whitespace only) → default to mrp1 if present
      - missing → mrp1
    """
    sel = (attrs.get("mrp0sel") or "").strip()
    if sel == "DEL":
        return None
    m = _MRP_SEL_NUMERIC.search(sel)
    idx = int(m.group(1)) if m else 1
    mrp_str = attrs.get(f"mrp{idx}")
    return parse_mrp(mrp_str)


def _localname(tag: str) -> str:
    """Strip XML namespace prefix from a tag name."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _extract_word(w_elem: ET.Element) -> dict:
    """Extract structured info for one <w> element."""
    attrs = dict(w_elem.attrib)
    trans = attrs.get("trans", "")

    sumerograms: list[str] = []
    akkadograms: list[str] = []
    determinatives: list[str] = []

    for child in w_elem.iter():
        if child is w_elem:
            continue
        tag = _localname(child.tag)
        text = (child.text or "").strip()
        if not text:
            continue
        if tag == "sGr":
            sumerograms.append(text)
        elif tag == "aGr":
            akkadograms.append(text)
        elif tag == "d":
            determinatives.append(text)

    mrp = select_mrp(attrs)

    return {
        "trans": trans,
        "mrp": mrp,
        "sumerograms": sumerograms,
        "akkadograms": akkadograms,
        "determinatives": determinatives,
        "has_sumerogram": bool(sumerograms),
        "has_akkadogram": bool(akkadograms),
    }


def parse_xml_string(xml_text: str, doc_id: str = "") -> dict:
    """Parse one TLHdig XML document. Returns {doc_id, lines, words}.

    Lines are reconstructed by grouping <w> elements between consecutive <lb>
    markers. The space character between transliterated tokens is preserved
    so the downstream FastText cleaner sees natural word boundaries.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        return {"doc_id": doc_id, "lines": [], "words": [], "error": str(e)}

    # Walk all elements in document order, grouping <w> by <lb> boundaries.
    lines: list[list[str]] = []
    words: list[dict] = []
    current_line: list[str] = []

    # Find docID from header if present
    if not doc_id:
        for elem in root.iter():
            if _localname(elem.tag) == "docID" and elem.text:
                doc_id = elem.text.strip()
                break

    for elem in root.iter():
        tag = _localname(elem.tag)
        if tag == "lb":
            if current_line:
                lines.append(current_line)
                current_line = []
        elif tag == "w":
            w = _extract_word(elem)
            if w["trans"]:
                current_line.append(w["trans"])
            words.append(w)

    if current_line:
        lines.append(current_line)

    return {
        "doc_id": doc_id,
        "lines": [" ".join(line) for line in lines if line],
        "words": words,
    }


def parse_xml_file(path: Path) -> dict:
    """Parse one TLHdig XML file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_xml_string(text, doc_id=path.stem)


def _word_to_lemma_record(w: dict) -> dict | None:
    """Convert parsed <w> to a flat lemma record compatible with anchor extraction.

    Returns None when the word has no usable mrp analysis (deleted, ambiguous).
    The 'gw' field carries the GERMAN gloss — downstream multilingual encoding
    handles the language bridge.
    """
    mrp = w.get("mrp")
    if not mrp:
        return None
    lemma = mrp.get("lemma", "")
    gloss = mrp.get("gloss", "")
    if not lemma:
        return None
    return {
        "form": w.get("trans", ""),
        "cf": lemma,
        "gw": gloss,
        "pos": mrp.get("features", "")[:60],  # truncated; full features captured separately
        "category": mrp.get("category", ""),
        "has_sumerogram": w.get("has_sumerogram", False),
        "has_akkadogram": w.get("has_akkadogram", False),
        "lang": "hit",
    }


def main(tlhdig_root: Path = TLHDIG_ROOT_DEFAULT):
    if not tlhdig_root.exists():
        raise SystemExit(
            f"TLHdig extract not found at {tlhdig_root}. "
            "Download from https://zenodo.org/records/15459134 and extract to that path."
        )

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    all_texts: list[dict] = []
    all_lemmas: list[dict] = []
    heterogram_counter: Counter[str] = Counter()
    sumerogram_counter: Counter[str] = Counter()
    akkadogram_counter: Counter[str] = Counter()

    xml_files = sorted(tlhdig_root.rglob("*.xml"))
    print(f"Parsing {len(xml_files)} XML files from {tlhdig_root}")

    for path in tqdm(xml_files, desc="TLHdig"):
        parsed = parse_xml_file(path)
        if parsed.get("error"):
            continue
        doc_id = parsed["doc_id"]
        if parsed["lines"]:
            all_texts.append({
                "p_number": doc_id,
                "lines": parsed["lines"],
                "source": "TLHdig",
            })
        for w in parsed["words"]:
            rec = _word_to_lemma_record(w)
            if rec:
                all_lemmas.append(rec)
            for s in w.get("sumerograms") or []:
                sumerogram_counter[s] += 1
                heterogram_counter[f"S:{s}"] += 1
            for a in w.get("akkadograms") or []:
                akkadogram_counter[a] += 1
                heterogram_counter[f"A:{a}"] += 1

    with open(DATA_RAW / "hittite_texts.json", "w", encoding="utf-8") as f:
        json.dump(all_texts, f, ensure_ascii=False, indent=2)
    with open(DATA_RAW / "hittite_lemmas.json", "w", encoding="utf-8") as f:
        json.dump(all_lemmas, f, ensure_ascii=False, indent=2)
    heterograms = {
        "sumerograms": dict(sumerogram_counter.most_common()),
        "akkadograms": dict(akkadogram_counter.most_common()),
    }
    with open(DATA_RAW / "hittite_heterograms.json", "w", encoding="utf-8") as f:
        json.dump(heterograms, f, ensure_ascii=False, indent=2)

    total_lines = sum(len(t["lines"]) for t in all_texts)
    unique_glosses = len({l["gw"] for l in all_lemmas if l["gw"]})
    print(f"\nTotal texts: {len(all_texts)}")
    print(f"Total lines: {total_lines}")
    print(f"Total Hittite lemma records: {len(all_lemmas)}")
    print(f"Unique German glosses: {unique_glosses}")
    print(f"Unique Sumerograms: {len(sumerogram_counter)}")
    print(f"Unique Akkadograms: {len(akkadogram_counter)}")


if __name__ == "__main__":
    main()
