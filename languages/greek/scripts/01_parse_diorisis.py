"""
Diorisis JSON Parser: convert TLG Beta Code-encoded Diorisis files to
hyper-glyphy lemma + text JSON.

Source: Diorisis Ancient Greek Corpus (Figshare 12251468), v1.51, 821 JSON
files, ~10M tokens spanning Homer to early 5th c. AD.

Each file contains `sentences`, each with `tokens`:
  - form: Beta Code surface form (e.g., "qeou\\s")
  - lemma.entry: Beta Code citation form (e.g., "qeo/s")
  - lemma.POS: noun/verb/adj/...
  - lemma.analyses: list of morphological analyses

Output:
  - data/raw/greek_texts.json:  [{p_number (= file basename), lines: [str], source}]
  - data/raw/greek_lemmas.json: [{form, cf, gw (EMPTY — Diorisis has no glosses), pos, lang: "grc"}]

Note: Diorisis does NOT carry English glosses. We pull glosses from a
separate LSJ-derived dictionary in 06_extract_anchors.py.

Beta Code is converted to Unicode polytonic Greek at parse time so the rest
of the pipeline (greek_normalize, FastText, etc.) sees real Greek text.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import betacode.conv as bc
from tqdm import tqdm

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
DIORISIS_ROOT_DEFAULT = Path("/tmp/diorisis")


def _convert_token_form(beta: str) -> str:
    """Beta Code -> Unicode polytonic Greek; preserve as-is if conversion fails."""
    if not beta:
        return ""
    try:
        return bc.beta_to_uni(beta)
    except Exception:
        return beta


def parse_file(path: Path) -> dict:
    """Parse one Diorisis JSON file. Returns {doc_id, author, title, lines, lemmas}."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    sentences = data.get("sentences", []) or []
    lemmas: list[dict] = []
    lines: list[str] = []

    # File names look like "Aeschylus (0085) - Agamemnon (005).json"
    stem = path.stem
    doc_id = stem
    author = ""
    title = ""
    if " - " in stem:
        parts = stem.split(" - ", 1)
        author = parts[0]
        title = parts[1]

    for sent in sentences:
        tokens = sent.get("tokens", []) or []
        line_tokens: list[str] = []
        for tok in tokens:
            if tok.get("type") != "word":
                continue
            form_beta = tok.get("form", "")
            lemma_obj = tok.get("lemma", {}) or {}
            cf_beta = lemma_obj.get("entry", "")
            pos = lemma_obj.get("POS", "")

            form_uni = _convert_token_form(form_beta)
            cf_uni = _convert_token_form(cf_beta)

            if form_uni:
                line_tokens.append(form_uni)

            if cf_uni or form_uni:
                lemmas.append({
                    "form": form_uni,
                    "cf": cf_uni,
                    "gw": "",  # Diorisis has no glosses; injected by 06 from LSJ
                    "pos": pos,
                    "lang": "grc",
                })

        if line_tokens:
            lines.append(" ".join(line_tokens))

    return {
        "doc_id": doc_id,
        "author": author,
        "title": title,
        "lines": lines,
        "lemmas": lemmas,
    }


def main(diorisis_root: Path = DIORISIS_ROOT_DEFAULT):
    if not diorisis_root.exists():
        raise SystemExit(
            f"Diorisis extract not found at {diorisis_root}. Download from "
            "https://figshare.com/articles/dataset/The_Diorisis_Ancient_Greek_Corpus_JSON_/12251468 "
            "and extract to that path."
        )

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    all_texts: list[dict] = []
    all_lemmas: list[dict] = []
    author_counter: Counter[str] = Counter()

    json_files = sorted(diorisis_root.rglob("*.json"))
    print(f"Parsing {len(json_files)} Diorisis JSON files from {diorisis_root}")

    for path in tqdm(json_files, desc="Diorisis"):
        try:
            parsed = parse_file(path)
        except Exception as e:
            print(f"  Skipped {path.name}: {e}")
            continue
        if parsed["lines"]:
            all_texts.append({
                "p_number": parsed["doc_id"],
                "lines": parsed["lines"],
                "author": parsed["author"],
                "title": parsed["title"],
                "source": "Diorisis",
            })
        all_lemmas.extend(parsed["lemmas"])
        if parsed["author"]:
            author_counter[parsed["author"]] += 1

    with open(DATA_RAW / "greek_texts.json", "w", encoding="utf-8") as f:
        json.dump(all_texts, f, ensure_ascii=False, indent=2)
    with open(DATA_RAW / "greek_lemmas.json", "w", encoding="utf-8") as f:
        json.dump(all_lemmas, f, ensure_ascii=False, indent=2)

    total_lines = sum(len(t["lines"]) for t in all_texts)
    unique_cf = len({l["cf"] for l in all_lemmas if l["cf"]})
    print(f"\nTotal texts: {len(all_texts)}")
    print(f"Total lines: {total_lines}")
    print(f"Total token-lemma records: {len(all_lemmas)}")
    print(f"Unique citation forms: {unique_cf}")
    print(f"Distinct authors: {len(author_counter)}")
    print(f"Top 10 authors by work count:")
    for author, n in author_counter.most_common(10):
        print(f"  {author}: {n}")


if __name__ == "__main__":
    main()
