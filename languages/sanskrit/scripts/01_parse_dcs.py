"""
DCS CoNLL-U Parser: convert Digital Corpus of Sanskrit conllu dumps to
hyper-glyphy lemma + text JSON.

Source: github.com/OliverHellwig/sanskrit (CC BY 4.0), dcs/data/conllu/files/
— one folder per text, one .conllu file per chapter. ~745k lines, ~5.46M words.

Chapter file headers:
  ## text: Aitareyopaniṣad
  ## text_id: 421
  ## chapter: AU, 1, 1
  ## chapter_id: 8816

Token lines are UD CoNLL-U: FORM (col 2) is the sandhi-resolved word form in
IAST; LEMMA (col 3) is the gold lemma. Multiword range lines (ID "1-2") and
empty nodes (ID "2.1") are skipped — their word lines carry the data. PUNCT
rows are skipped. Malformed token lines (fewer than 10 columns) are counted
and reported as parse loss, never silently dropped.

Output:
  - data/raw/sanskrit_texts.json:
      [{p_number: "dcs-<text_id>-<chapter_id>", text_name, chapter,
        lines: [str], source: "DCS"}]
  - data/raw/sanskrit_lemmas.json:
      [{form, cf, gw (EMPTY — glosses joined in 06 from MW), pos, lang: "san"}]
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from tqdm import tqdm

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
DCS_FILES_DEFAULT = DATA_RAW / "dcs" / "dcs" / "data" / "conllu" / "files"

_HEADER_RE = re.compile(r"##\s+(text|text_id|chapter|chapter_id):\s*(.*)")


def parse_file(path: Path) -> dict:
    """Parse one DCS chapter .conllu file."""
    meta = {"text": "", "text_id": "", "chapter": "", "chapter_id": ""}
    lines: list[str] = []
    lemmas: list[dict] = []
    token_lines = 0
    bad_lines = 0
    sent_tokens: list[str] = []

    def flush():
        nonlocal sent_tokens
        if sent_tokens:
            lines.append(" ".join(sent_tokens))
            sent_tokens = []

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("##"):
                m = _HEADER_RE.match(line)
                if m:
                    meta[m.group(1)] = m.group(2).strip()
                continue
            if not line.strip():
                flush()
                continue
            if line.startswith("#"):
                continue
            if not line[0].isdigit():
                continue
            token_lines += 1
            cols = line.split("\t")
            if len(cols) < 10:
                bad_lines += 1
                continue
            tok_id = cols[0]
            if "-" in tok_id or "." in tok_id:
                continue  # multiword range / empty node
            form, lemma, upos = cols[1], cols[2], cols[3]
            if upos == "PUNCT" or form in ("", "_"):
                continue
            sent_tokens.append(form)
            lemmas.append({
                "form": form,
                "cf": lemma if lemma != "_" else "",
                "gw": "",  # DCS has no glosses; injected by 06 from MW
                "pos": upos,
                "lang": "san",
            })
    flush()
    return {**meta, "lines": lines, "lemmas": lemmas,
            "token_lines": token_lines, "bad_lines": bad_lines}


def main(dcs_files: Path = DCS_FILES_DEFAULT):
    if not dcs_files.exists():
        raise SystemExit(
            f"DCS conllu files not found at {dcs_files}. Run the documented "
            "fetch step (languages/sanskrit/README.md): sparse clone of "
            "https://github.com/OliverHellwig/sanskrit.git into data/raw/dcs."
        )
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    all_texts: list[dict] = []
    all_lemmas: list[dict] = []
    token_lines = 0
    bad_lines = 0

    conllu_files = sorted(dcs_files.rglob("*.conllu"))
    print(f"Parsing {len(conllu_files)} DCS conllu files from {dcs_files}")

    for path in tqdm(conllu_files, desc="DCS"):
        parsed = parse_file(path)
        token_lines += parsed["token_lines"]
        bad_lines += parsed["bad_lines"]
        if parsed["lines"]:
            all_texts.append({
                "p_number": f"dcs-{parsed['text_id']}-{parsed['chapter_id']}",
                "text_name": parsed["text"],
                "chapter": parsed["chapter"],
                "lines": parsed["lines"],
                "source": "DCS",
            })
        all_lemmas.extend(parsed["lemmas"])

    with open(DATA_RAW / "sanskrit_texts.json", "w", encoding="utf-8") as f:
        json.dump(all_texts, f, ensure_ascii=False, indent=2)
    with open(DATA_RAW / "sanskrit_lemmas.json", "w", encoding="utf-8") as f:
        json.dump(all_lemmas, f, ensure_ascii=False, indent=2)

    total_lines = sum(len(t["lines"]) for t in all_texts)
    unique_cf = len({l["cf"] for l in all_lemmas if l["cf"]})
    loss_pct = 100.0 * bad_lines / max(1, token_lines)
    print(f"\nTotal chapter files: {len(all_texts)}")
    print(f"Total lines: {total_lines}")
    print(f"Total token-lemma records: {len(all_lemmas)}")
    print(f"Unique lemmas (cf): {unique_cf}")
    print(f"Parse loss: {bad_lines}/{token_lines} token lines ({loss_pct:.3f}%)")


if __name__ == "__main__":
    main()
