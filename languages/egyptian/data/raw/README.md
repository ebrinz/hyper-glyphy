# languages/egyptian/data/raw

This directory is currently empty for the Egyptian slot (gitignored except
this file). Per `languages/egyptian/README.md`, pipeline scripts 01-05
(corpus building from primary sources) are **not yet ported** — the cleaned
corpus and anchors were migrated pre-built from the predecessor `heiroglyphy`
V15 project directly into `data/processed/`, not `data/raw/`.

## Source

`06_extract_anchors.py` reads `data/processed/english_anchors.json` — 8,541
Egyptian-English pairs sourced (per the script's docstring) from TLA, Ramses,
and BBAW. No fetch script exists in this repo for the original TLA/Ramses/BBAW
data; if reconstructing from scratch, see those projects directly:

- TLA — Thesaurus Linguae Aegyptiae (`https://thesaurus-linguae-aegyptiae.de/`)
- Ramses (`http://ramses.ulg.ac.be/`)
- BBAW (Berlin-Brandenburgische Akademie der Wissenschaften)

License: not stated in-repo; consult each project directly.

## If a raw corpus is ever added

There is no default path or fetch command for it yet — see
`06_extract_anchors.py` (reads `data/processed/english_anchors.json`) and
`languages/egyptian/README.md` for the current pipeline entry point.
