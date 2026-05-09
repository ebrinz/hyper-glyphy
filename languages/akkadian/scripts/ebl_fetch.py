"""
eBL (electronic Babylonian Library) lemma fetcher.

Pulls the full dictionary from https://www.ebl.lmu.de/api/dictionary, caches it
to data/dictionaries/ebl_lemmas.json, and exposes filtering helpers used by
06_extract_anchors.py.

Each eBL entry includes:
  - lemma: list[str] (citation form)
  - guideWord: short English gloss
  - periodAttestation: list[str] (e.g., ["Old Babylonian", ...])
  - logograms: list[dict] with 'logogram' key
"""
import json
from pathlib import Path

import requests

DATA_DICTS = Path(__file__).parent.parent / "data" / "dictionaries"
EBL_API = "https://www.ebl.lmu.de/api/dictionary"
CACHE_PATH = DATA_DICTS / "ebl_lemmas.json"


def fetch_ebl_lemmas(force: bool = False) -> list[dict]:
    """Fetch all eBL lemmas, caching to disk. Returns the parsed list."""
    DATA_DICTS.mkdir(parents=True, exist_ok=True)
    if CACHE_PATH.exists() and not force:
        with open(CACHE_PATH) as f:
            return json.load(f)
    response = requests.get(EBL_API, timeout=600)
    response.raise_for_status()
    data = response.json()
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data


def filter_ob_lemmas(lemmas: list[dict]) -> list[dict]:
    """Keep entries with Old Babylonian attestation and a non-empty guideWord.

    Returns flat dicts with fields: lemma, gloss, period, logograms, raw.
    """
    out = []
    for entry in lemmas:
        periods = entry.get("periodAttestation") or []
        if not any("Old Babylonian" in p for p in periods):
            continue
        gw = (entry.get("guideWord") or "").strip()
        if not gw:
            continue
        lemma_parts = entry.get("lemma") or []
        if not lemma_parts:
            continue
        out.append({
            "lemma": lemma_parts[0],
            "gloss": gw,
            "period": periods,
            "logograms": [l.get("logogram") for l in (entry.get("logograms") or []) if l.get("logogram")],
            "raw": entry,
        })
    return out


def extract_surface_forms(entry: dict) -> list[str]:
    """Return citation form + logogram surface forms for a flattened entry."""
    forms = []
    if entry.get("lemma"):
        forms.append(entry["lemma"])
    forms.extend(entry.get("logograms") or [])
    return forms


def main():
    print(f"Fetching eBL lemmas (cache: {CACHE_PATH})...")
    lemmas = fetch_ebl_lemmas()
    ob = filter_ob_lemmas(lemmas)
    print(f"Total eBL entries: {len(lemmas)}")
    print(f"OB-attested with gloss: {len(ob)}")


if __name__ == "__main__":
    main()
