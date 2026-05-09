"""
ORACC OB Akkadian Scraper: Download and parse OB-period Akkadian text JSON archives.

Source: http://oracc.museum.upenn.edu/{PROJECT}/json.zip
Format: Hierarchical CDL JSON with lemmatized Akkadian words.

Lemma node 'f' key contains:
  - form: surface transliteration
  - cf: citation form (dictionary headword)
  - gw: guide word (English gloss)
  - pos: part of speech
  - lang: language code (akk, akk-x-stdbab, akk-x-oldbab, etc.)

This scraper accepts any 'akk' prefix on the lang code so OB-marked dialect
variants are captured. Period filtering is applied later in 06_extract_anchors.py
via the eBL period flag.
"""
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"

# Seed list — OB-period-relevant ORACC sub-projects.
# To extend: append a project slug, re-run.
ORACC_PROJECTS = [
    "hbtin",         # Hethitologie Portal index of OB texts (where ATF-available)
    "saao/saa",      # State Archives of Assyria (NA — kept for SB/canonical descendants)
    "blms",          # Bilingual literary; OB recensions of canonical texts
    "rinap/rinap1",  # Royal Inscriptions of the Neo-Assyrian Period (descendant tradition)
]

ORACC_BASE_URL = "https://oracc.museum.upenn.edu/json"


def download_project_json(project: str, output_dir: Path) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"oracc_{project.replace('/', '_')}.zip"

    if zip_path.exists():
        print(f"  Already downloaded: {zip_path}")
        return zip_path

    slug = project.replace("/", "-")
    url = f"{ORACC_BASE_URL}/{slug}.zip"
    try:
        response = requests.get(url, timeout=600, verify=False, stream=True)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(zip_path, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc=f"  {slug}", leave=False) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return zip_path
    except requests.RequestException as e:
        print(f"  Failed to download {project}: {e}")
        return None


def _walk_cdl(node: Any, lemmas: list[dict], line_words: list[list[str]], current_line: list[str]) -> None:
    if isinstance(node, dict):
        if "f" in node:
            f = node["f"]
            lang = f.get("lang", "")
            if lang.startswith("akk"):
                lemma = {
                    "form": f.get("form", ""),
                    "cf": f.get("cf", ""),
                    "gw": f.get("gw", ""),
                    "pos": f.get("pos", ""),
                    "norm": f.get("norm", ""),
                    "lang": lang,
                }
                if lemma["form"]:
                    lemmas.append(lemma)
                    current_line.append(lemma["form"])

        if node.get("ftype") == "line-start" or node.get("type") == "line-start":
            if current_line:
                line_words.append(list(current_line))
                current_line.clear()

        if "cdl" in node:
            for child in node["cdl"]:
                _walk_cdl(child, lemmas, line_words, current_line)

    elif isinstance(node, list):
        for child in node:
            _walk_cdl(child, lemmas, line_words, current_line)


def extract_lemmas(text_json: dict) -> list[dict]:
    lemmas, line_words, current_line = [], [], []
    _walk_cdl(text_json.get("cdl", []), lemmas, line_words, current_line)
    return lemmas


def extract_lines(text_json: dict) -> list[str]:
    lemmas, line_words, current_line = [], [], []
    _walk_cdl(text_json.get("cdl", []), lemmas, line_words, current_line)
    if current_line:
        line_words.append(current_line)
    return [" ".join(words) for words in line_words if words]


def parse_project_zip(zip_path: Path) -> tuple[list[dict], list[dict]]:
    all_lemmas, all_texts = [], []
    with zipfile.ZipFile(zip_path) as zf:
        json_files = [n for n in zf.namelist() if "corpusjson" in n and n.endswith(".json")]
        for name in json_files:
            try:
                data = json.loads(zf.read(name))
            except (json.JSONDecodeError, KeyError):
                continue
            p_number = Path(name).stem
            lemmas = extract_lemmas(data)
            lines = extract_lines(data)
            if lemmas:
                all_lemmas.extend(lemmas)
            if lines:
                all_texts.append({"p_number": p_number, "lines": lines, "source": "ORACC"})
    return all_lemmas, all_texts


def save_texts(texts: list[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(texts)} texts to {output_path}")


def main():
    out_dir = DATA_RAW / "ob_literary"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_texts, all_lemmas = [], []

    for project in tqdm(ORACC_PROJECTS, desc="Downloading OB-relevant ORACC projects"):
        print(f"\nProcessing {project}...")
        zip_path = download_project_json(project, out_dir)
        if zip_path is None:
            continue
        lemmas, texts = parse_project_zip(zip_path)
        all_lemmas.extend(lemmas)
        all_texts.extend(texts)
        print(f"  {len(texts)} texts, {len(lemmas)} Akkadian lemmas")

    save_texts(all_texts, str(DATA_RAW / "ob_literary_texts.json"))
    save_texts(all_lemmas, str(DATA_RAW / "ob_literary_lemmas.json"))

    total_lines = sum(len(t["lines"]) for t in all_texts)
    unique_glosses = len({l["gw"] for l in all_lemmas if l["gw"]})
    print(f"\nTotal texts: {len(all_texts)}")
    print(f"Total lines: {total_lines}")
    print(f"Total Akkadian lemmas: {len(all_lemmas)}")
    print(f"Unique English glosses: {unique_glosses}")


if __name__ == "__main__":
    main()
