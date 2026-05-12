"""
LSJ Parser: extract Greek-English glosses from Perseus LSJ XML.

Source: Perseus Digital Library lexica repo
(github.com/PerseusDL/lexica, CTS_XML_TEI/perseus/pdllex/grc/lsj/),
28 XML files totaling ~282MB, ~115k entries.

For each <entryFree key="..."> element:
  - key: Beta Code lemma (e.g., "a)gaqo/s")
  - <tr> children: short English glosses

Output: data/dictionaries/lsj_glosses.json
  [{lemma_uni, lemma_norm, glosses: [...], gloss_first}]
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import betacode.conv as bc
from tqdm import tqdm

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from languages.greek.scripts.greek_normalize import normalize_greek_token  # noqa: E402

DATA_DICTS = Path(__file__).parent.parent / "data" / "dictionaries"
LSJ_ROOT_DEFAULT = Path("/tmp/lsj")

_TRAILING_PUNCT = re.compile(r"[\s,.;:—\-]+$")


def _localname(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _clean_gloss(gloss: str) -> str:
    """Trim trailing comma/period/dash whitespace from a <tr> element's text."""
    if not gloss:
        return ""
    return _TRAILING_PUNCT.sub("", gloss.strip())


def _convert_key(beta: str) -> str:
    if not beta:
        return ""
    try:
        return bc.beta_to_uni(beta)
    except Exception:
        return beta


def parse_lsj_file(path: Path) -> list[dict]:
    """Parse one LSJ XML file. Returns list of entry dicts."""
    entries: list[dict] = []
    try:
        tree = ET.parse(path)
    except ET.ParseError as e:
        print(f"  Skipped {path.name}: {e}")
        return entries
    root = tree.getroot()
    for e in root.iter():
        if _localname(e.tag) != "entryFree":
            continue
        key_beta = e.attrib.get("key", "")
        if not key_beta:
            continue
        glosses = []
        for t in e.iter():
            if _localname(t.tag) == "tr":
                cleaned = _clean_gloss(t.text or "")
                if cleaned:
                    glosses.append(cleaned)
        if not glosses:
            continue
        lemma_uni = _convert_key(key_beta)
        lemma_norm = normalize_greek_token(lemma_uni)
        if not lemma_norm:
            continue
        entries.append({
            "lemma_beta": key_beta,
            "lemma_uni": lemma_uni,
            "lemma_norm": lemma_norm,
            "glosses": glosses,
            "gloss_first": glosses[0],
        })
    return entries


def main(lsj_root: Path = LSJ_ROOT_DEFAULT):
    if not lsj_root.exists():
        raise SystemExit(
            f"LSJ XML directory not found at {lsj_root}. Run the download script first."
        )
    DATA_DICTS.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(lsj_root.glob("*.xml"))
    print(f"Parsing {len(xml_files)} LSJ XML files")

    all_entries: list[dict] = []
    for path in tqdm(xml_files, desc="LSJ files"):
        all_entries.extend(parse_lsj_file(path))

    # Deduplicate by normalized lemma — keep first occurrence (usually the main entry)
    seen: dict[str, dict] = {}
    for entry in all_entries:
        key = entry["lemma_norm"]
        if key not in seen:
            seen[key] = entry

    deduped = list(seen.values())
    output_path = DATA_DICTS / "lsj_glosses.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)

    print(f"\nTotal entries parsed: {len(all_entries)}")
    print(f"After dedup by lemma_norm: {len(deduped)}")
    print(f"Saved to: {output_path}")

    # Sample
    print("\nFirst 10 glosses:")
    for e in deduped[:10]:
        print(f"  {e['lemma_uni']:>15s}  [{e['lemma_norm']:>12s}]  ->  {e['gloss_first']!r}")


if __name__ == "__main__":
    main()
