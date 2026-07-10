"""
Fetch Sumerian incantation texts from ORACC blms project
(Bilinguals in Late Mesopotamian Scholarship — bilingual Sumerian-Akkadian
incantation/prayer tablets including the Udug-hul / Utukkū Lemnūtu series).

Source: https://oracc.museum.upenn.edu/json/blms.zip
Output: languages/sumerian/data/processed/incantation_docs.json
Format: [{"doc_id": str, "tokens": [str, ...]}, ...]

Only docs with >=30 in-vocab tokens (against fused_embeddings_1536d.npz) are kept.
Overall vocab hit rate must be >=40% or the run is blocked.

Network fetch is NOT unit-tested; parsing/normalization functions are.
"""
import json
import zipfile
from pathlib import Path
from typing import Any

import sys

import numpy as np
import requests
from tqdm import tqdm

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

BLMS_PROJECT = "blms"
ORACC_BASE_URL = "https://oracc.museum.upenn.edu/json"
# Genres (lowercase) that qualify as incantation content
INCANT_GENRES = ("incantation", "prayer")

# Preferred ZIP location for sumerian; fall back to akkadian's already-downloaded copy
_SUMERIAN_RAW = Path(__file__).parent.parent / "data" / "raw"
_AKKADIAN_BLMS = (_ROOT / "languages" / "akkadian" / "data" / "raw"
                  / "ob_literary" / "oracc_blms.zip")
BLMS_ZIP_PATH = _SUMERIAN_RAW / "incantations" / "oracc_blms.zip"

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "processed" / "incantation_docs.json"
VOCAB_PATH = Path(__file__).parent.parent / "models" / "fused_embeddings_1536d.npz"

MIN_IN_VOCAB_TOKENS = 30
MIN_HIT_RATE_PCT = 40.0


# ---------------------------------------------------------------------------
# Testable parsing/normalization functions
# ---------------------------------------------------------------------------

def is_incantation(meta: dict) -> bool:
    """True when catalogue genre indicates incantation or prayer content."""
    genre = meta.get("genre", "").lower()
    return bool(genre) and any(g in genre for g in INCANT_GENRES)


def _walk_cdl(node: Any, tokens: list) -> None:
    """Recursively collect Sumerian form tokens from ORACC CDL JSON tree."""
    if isinstance(node, dict):
        if "f" in node:
            f = node["f"]
            if f.get("lang", "").startswith("sux"):
                form = f.get("form", "")
                if form:
                    tokens.append(form)
        for child in node.get("cdl", []):
            _walk_cdl(child, tokens)
    elif isinstance(node, list):
        for child in node:
            _walk_cdl(child, tokens)


def extract_sux_tokens(text_json: dict) -> list:
    """Extract raw Sumerian form tokens from ORACC CDL JSON."""
    tokens: list = []
    _walk_cdl(text_json.get("cdl", []), tokens)
    return tokens


def load_catalogue(zip_path: Path) -> dict:
    """Return blms catalogue members dict keyed by text id, or {} if absent."""
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith("catalogue.json"):
                return json.loads(zf.read(name)).get("members", {})
    return {}


def parse_incantation_zip(zip_path: Path, members: dict) -> list:
    """Parse blms ZIP; return list of {doc_id, tokens} for incantation texts.

    If members is non-empty, only texts catalogued as incantation/prayer are kept.
    Tokens are raw forms (not yet normalized) — caller must normalize.
    """
    incant_ids = {k for k, v in members.items() if is_incantation(v)} if members else None

    docs = []
    with zipfile.ZipFile(zip_path) as zf:
        corpus_files = [n for n in zf.namelist()
                        if "corpusjson" in n and n.endswith(".json")]
        for name in corpus_files:
            doc_id = Path(name).stem
            if incant_ids is not None and doc_id not in incant_ids:
                continue
            try:
                data = json.loads(zf.read(name))
            except (json.JSONDecodeError, KeyError):
                continue
            raw = extract_sux_tokens(data)
            if raw:
                docs.append({"doc_id": doc_id, "raw_tokens": raw})
    return docs


def normalize_docs(raw_docs: list) -> list:
    """Apply normalize_sumerian_token to each doc's raw tokens; drop empty tokens."""
    from languages.sumerian.scripts.sumerian_normalize import normalize_sumerian_token

    out = []
    for doc in raw_docs:
        tokens = [t for t in (normalize_sumerian_token(r) for r in doc["raw_tokens"]) if t]
        if tokens:
            out.append({"doc_id": doc["doc_id"], "tokens": tokens})
    return out


def compute_hit_stats(docs: list, vocab: dict) -> tuple:
    """Return (overall_rate_pct, per_doc_stats, kept_docs).

    kept_docs: docs with >=MIN_IN_VOCAB_TOKENS in-vocab tokens.
    per_doc_stats: list of {doc_id, n_tokens, n_in_vocab, hit_rate_pct}.
    """
    total, in_vocab_total = 0, 0
    per_doc = []
    kept = []
    for doc in docs:
        toks = doc["tokens"]
        iv = sum(1 for t in toks if t in vocab)
        total += len(toks)
        in_vocab_total += iv
        hit_pct = iv / max(len(toks), 1) * 100
        per_doc.append({"doc_id": doc["doc_id"], "n_tokens": len(toks),
                        "n_in_vocab": iv, "hit_rate_pct": round(hit_pct, 1)})
        if iv >= MIN_IN_VOCAB_TOKENS:
            kept.append(doc)
    overall = in_vocab_total / max(total, 1) * 100
    return overall, per_doc, kept


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def locate_blms_zip() -> Path:
    """Return path to blms ZIP: preferred sumerian location, then akkadian fallback."""
    if BLMS_ZIP_PATH.exists():
        return BLMS_ZIP_PATH
    if _AKKADIAN_BLMS.exists():
        print(f"  Using existing blms ZIP from akkadian data: {_AKKADIAN_BLMS}")
        return _AKKADIAN_BLMS
    return None


def download_blms_zip() -> Path:
    """Download blms ZIP to the sumerian incantations data dir."""
    BLMS_ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    slug = BLMS_PROJECT.replace("/", "-")
    url = f"{ORACC_BASE_URL}/{slug}.zip"
    print(f"Downloading {url} ...")
    resp = requests.get(url, timeout=600, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(BLMS_ZIP_PATH, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc="blms") as pbar:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
                pbar.update(len(chunk))
    return BLMS_ZIP_PATH


def main():
    zip_path = locate_blms_zip()
    if zip_path is None:
        zip_path = download_blms_zip()
    print(f"blms ZIP: {zip_path}")

    members = load_catalogue(zip_path)
    n_incant = sum(1 for v in members.values() if is_incantation(v))
    print(f"Catalogue members: {len(members)} total, {n_incant} incantation/prayer")

    raw_docs = parse_incantation_zip(zip_path, members)
    print(f"Corpus JSON files with Sumerian tokens (incantation): {len(raw_docs)}")

    docs = normalize_docs(raw_docs)
    print(f"Docs after normalization: {len(docs)}")

    data = np.load(str(VOCAB_PATH), allow_pickle=True)
    vocab = {str(w): i for i, w in enumerate(data["vocab"])}
    overall_rate, per_doc, kept = compute_hit_stats(docs, vocab)

    total_toks = sum(d["n_tokens"] for d in per_doc)
    total_iv = sum(d["n_in_vocab"] for d in per_doc)
    print(f"Vocab hit rate: {overall_rate:.1f}% ({total_iv}/{total_toks} tokens)")
    print(f"Docs kept (>={MIN_IN_VOCAB_TOKENS} in-vocab tokens): {len(kept)}/{len(docs)}")

    if overall_rate < MIN_HIT_RATE_PCT:
        print(f"BLOCKED: vocab hit rate {overall_rate:.1f}% < {MIN_HIT_RATE_PCT}% threshold")
        return

    if not kept:
        print(f"BLOCKED: no docs with >={MIN_IN_VOCAB_TOKENS} in-vocab tokens")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(kept)} incantation docs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
