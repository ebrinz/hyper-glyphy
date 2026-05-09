"""
DCCLT Scraper: Download Digital Corpus of Cuneiform Lexical Texts and parse
both running text (for Akkadian FastText) and Sumerian↔Akkadian word pairs (for
the v2 cross-lingual bridge experiment).

Source: https://oracc.museum.upenn.edu/json/dcclt.zip
Output:
  - data/raw/dcclt/                              (downloaded zip)
  - data/raw/dcclt_texts.json                    (Akkadian-side running text)
  - data/raw/dcclt_lemmas.json                   (Akkadian lemmas — fallback anchors)
  - data/processed/sumerian_akkadian_pairs.jsonl (parsed sux↔akk pairs)
"""
import importlib.util
import json
from pathlib import Path
from typing import Any
import zipfile

from tqdm import tqdm

_OB_SCRAPER_PATH = Path(__file__).parent / "01_scrape_oracc_ob.py"
_spec = importlib.util.spec_from_file_location("ob_scraper", _OB_SCRAPER_PATH)
_ob = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ob)

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"

DCCLT_PROJECT = "dcclt"


def _walk_lines(node: Any, lines: list[list[dict]]) -> None:
    current: list[dict] = []
    _recurse(node, lines, current)
    if current:
        lines.append(current)


def _recurse(node: Any, lines: list[list[dict]], current: list[dict]) -> None:
    if isinstance(node, dict):
        if "f" in node:
            f = node["f"]
            current.append({
                "lang": f.get("lang", ""),
                "cf": f.get("cf", ""),
                "form": f.get("form", ""),
                "gw": f.get("gw", ""),
            })
        if node.get("ftype") == "line-start" or node.get("type") == "line-start":
            if current:
                lines.append(list(current))
                current.clear()
        if "cdl" in node:
            for child in node["cdl"]:
                _recurse(child, lines, current)
    elif isinstance(node, list):
        for child in node:
            _recurse(child, lines, current)


def extract_pairs(text_json: dict) -> list[dict]:
    """Emit Sumerian↔Akkadian word pairs from a DCCLT text JSON.

    A pair is emitted when one line contains BOTH a sux lemma and an akk lemma.
    """
    lines: list[list[dict]] = []
    _walk_lines(text_json.get("cdl", []), lines)
    pairs: list[dict] = []
    for line in lines:
        sux = next((w for w in line if w["lang"].startswith("sux")), None)
        akk = next((w for w in line if w["lang"].startswith("akk")), None)
        if sux and akk:
            pairs.append({
                "sumerian_cf": sux["cf"],
                "sumerian_form": sux["form"],
                "sumerian_gw": sux["gw"],
                "akkadian_cf": akk["cf"],
                "akkadian_form": akk["form"],
                "akkadian_gw": akk["gw"],
            })
    return pairs


def parse_dcclt_zip(zip_path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    all_lemmas, all_texts, all_pairs = [], [], []
    with zipfile.ZipFile(zip_path) as zf:
        json_files = [n for n in zf.namelist() if "corpusjson" in n and n.endswith(".json")]
        for name in tqdm(json_files, desc="  DCCLT files", leave=False):
            try:
                data = json.loads(zf.read(name))
            except (json.JSONDecodeError, KeyError):
                continue
            p_number = Path(name).stem
            lemmas = _ob.extract_lemmas(data)
            lines = _ob.extract_lines(data)
            pairs = extract_pairs(data)
            if lemmas:
                all_lemmas.extend(lemmas)
            if lines:
                all_texts.append({"p_number": p_number, "lines": lines, "source": "DCCLT"})
            all_pairs.extend(pairs)
    return all_lemmas, all_texts, all_pairs


def main():
    out_dir = DATA_RAW / "dcclt"
    out_dir.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    zip_path = _ob.download_project_json(DCCLT_PROJECT, out_dir)
    if zip_path is None:
        print("Failed to download DCCLT")
        return

    lemmas, texts, pairs = parse_dcclt_zip(zip_path)

    _ob.save_texts(texts, str(DATA_RAW / "dcclt_texts.json"))
    _ob.save_texts(lemmas, str(DATA_RAW / "dcclt_lemmas.json"))

    pairs_path = DATA_PROCESSED / "sumerian_akkadian_pairs.jsonl"
    with open(pairs_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {len(pairs)} sux↔akk pairs to {pairs_path}")

    print(f"\nDCCLT akkadian texts: {len(texts)}")
    print(f"DCCLT akkadian lemmas: {len(lemmas)}")
    print(f"DCCLT bridge pairs: {len(pairs)}")
