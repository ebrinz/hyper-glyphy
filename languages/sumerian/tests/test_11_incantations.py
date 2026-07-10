"""
Tests for 11_fetch_incantations.py — parsing and normalization logic only.
Network fetch and file I/O are not tested here.
"""
import io
import json
import zipfile

import pytest


# ---------------------------------------------------------------------------
# is_incantation
# ---------------------------------------------------------------------------

def test_is_incantation_prayer():
    from languages.sumerian.scripts.fetch_incantations_11 import is_incantation
    assert is_incantation({"genre": "Prayer/Incantation"}) is True


def test_is_incantation_incantation_only():
    from languages.sumerian.scripts.fetch_incantations_11 import is_incantation
    assert is_incantation({"genre": "Incantation"}) is True


def test_is_incantation_incantation_ritual():
    from languages.sumerian.scripts.fetch_incantations_11 import is_incantation
    assert is_incantation({"genre": "Incantation-Ritual"}) is True


def test_is_incantation_ritual_only_is_false():
    from languages.sumerian.scripts.fetch_incantations_11 import is_incantation
    assert is_incantation({"genre": "Ritual"}) is False


def test_is_incantation_literary_is_false():
    from languages.sumerian.scripts.fetch_incantations_11 import is_incantation
    assert is_incantation({"genre": "Literary"}) is False


def test_is_incantation_no_genre_is_false():
    from languages.sumerian.scripts.fetch_incantations_11 import is_incantation
    assert is_incantation({}) is False


def test_is_incantation_case_insensitive():
    from languages.sumerian.scripts.fetch_incantations_11 import is_incantation
    assert is_incantation({"genre": "PRAYER/INCANTATION"}) is True


# ---------------------------------------------------------------------------
# extract_sux_tokens
# ---------------------------------------------------------------------------

_SAMPLE_CDL = {
    "cdl": [
        {
            "node": "c",
            "cdl": [
                {
                    "node": "l",
                    "f": {"lang": "sux", "form": "an", "cf": "an", "gw": "sky"},
                },
                {
                    "node": "l",
                    "f": {"lang": "sux", "form": "ki", "cf": "ki", "gw": "earth"},
                },
                # Akkadian word — should be ignored
                {
                    "node": "l",
                    "f": {"lang": "akk", "form": "ilu", "cf": "ilu", "gw": "god"},
                },
            ],
        }
    ]
}


def test_extract_sux_tokens_basic():
    from languages.sumerian.scripts.fetch_incantations_11 import extract_sux_tokens
    tokens = extract_sux_tokens(_SAMPLE_CDL)
    assert tokens == ["an", "ki"]


def test_extract_sux_tokens_skips_akkadian():
    from languages.sumerian.scripts.fetch_incantations_11 import extract_sux_tokens
    tokens = extract_sux_tokens(_SAMPLE_CDL)
    assert "ilu" not in tokens


def test_extract_sux_tokens_empty_json():
    from languages.sumerian.scripts.fetch_incantations_11 import extract_sux_tokens
    assert extract_sux_tokens({}) == []


def test_extract_sux_tokens_nested_cdl():
    from languages.sumerian.scripts.fetch_incantations_11 import extract_sux_tokens
    nested = {
        "cdl": [
            {"node": "c", "cdl": [
                {"node": "c", "cdl": [
                    {"node": "l", "f": {"lang": "sux", "form": "lugal", "cf": "lugal"}}
                ]}
            ]}
        ]
    }
    assert extract_sux_tokens(nested) == ["lugal"]


def test_extract_sux_tokens_skips_empty_form():
    from languages.sumerian.scripts.fetch_incantations_11 import extract_sux_tokens
    text = {"cdl": [{"node": "l", "f": {"lang": "sux", "form": ""}}]}
    assert extract_sux_tokens(text) == []


# ---------------------------------------------------------------------------
# normalize_docs
# ---------------------------------------------------------------------------

def test_normalize_docs_applies_sumerian_normalize():
    from languages.sumerian.scripts.fetch_incantations_11 import normalize_docs
    raw = [{"doc_id": "P001", "raw_tokens": ["šar₂", "an-na"]}]
    result = normalize_docs(raw)
    assert len(result) == 1
    assert result[0]["doc_id"] == "P001"
    # š -> sz, subscript 2 -> 2, hyphen dropped
    tokens = result[0]["tokens"]
    assert "szar2" in tokens
    assert "anna" in tokens


def test_normalize_docs_drops_empty_tokens():
    from languages.sumerian.scripts.fetch_incantations_11 import normalize_docs
    raw = [{"doc_id": "P002", "raw_tokens": ["", "  ", "lugal"]}]
    result = normalize_docs(raw)
    assert result[0]["tokens"] == ["lugal"]


def test_normalize_docs_skips_all_empty():
    from languages.sumerian.scripts.fetch_incantations_11 import normalize_docs
    raw = [{"doc_id": "P003", "raw_tokens": ["", "  "]}]
    result = normalize_docs(raw)
    assert result == []


# ---------------------------------------------------------------------------
# compute_hit_stats
# ---------------------------------------------------------------------------

def test_compute_hit_stats_all_in_vocab():
    from languages.sumerian.scripts.fetch_incantations_11 import compute_hit_stats
    vocab = {"a": 0, "b": 1, "c": 2}
    docs = [{"doc_id": "D1", "tokens": ["a"] * 50}]
    rate, per_doc, kept = compute_hit_stats(docs, vocab)
    assert rate == pytest.approx(100.0)
    assert len(kept) == 1
    assert per_doc[0]["n_in_vocab"] == 50


def test_compute_hit_stats_none_in_vocab():
    from languages.sumerian.scripts.fetch_incantations_11 import compute_hit_stats
    vocab = {"x": 0}
    docs = [{"doc_id": "D1", "tokens": ["zzz"] * 50}]
    rate, per_doc, kept = compute_hit_stats(docs, vocab)
    assert rate == pytest.approx(0.0)
    assert kept == []


def test_compute_hit_stats_min_in_vocab_threshold():
    from languages.sumerian.scripts.fetch_incantations_11 import compute_hit_stats, MIN_IN_VOCAB_TOKENS
    vocab = {"a": 0}
    # Exactly at threshold
    docs = [{"doc_id": "D1", "tokens": ["a"] * MIN_IN_VOCAB_TOKENS + ["zzz"] * 10}]
    rate, per_doc, kept = compute_hit_stats(docs, vocab)
    assert len(kept) == 1
    # One below threshold
    docs2 = [{"doc_id": "D2", "tokens": ["a"] * (MIN_IN_VOCAB_TOKENS - 1)}]
    _, _, kept2 = compute_hit_stats(docs2, vocab)
    assert kept2 == []


def test_compute_hit_stats_multiple_docs():
    from languages.sumerian.scripts.fetch_incantations_11 import compute_hit_stats
    vocab = {"a": 0}
    docs = [
        {"doc_id": "D1", "tokens": ["a"] * 40},  # 40 in-vocab → kept
        {"doc_id": "D2", "tokens": ["z"] * 40},  # 0 in-vocab → dropped
    ]
    rate, per_doc, kept = compute_hit_stats(docs, vocab)
    assert len(kept) == 1
    assert kept[0]["doc_id"] == "D1"
    assert rate == pytest.approx(50.0)  # 40/(40+40)


# ---------------------------------------------------------------------------
# parse_incantation_zip (uses in-memory ZIP fixture)
# ---------------------------------------------------------------------------

def _make_blms_zip(texts: list) -> bytes:
    """Build a minimal blms-format ZIP in memory from a list of (p_id, genre, cdl_words)."""
    buf = io.BytesIO()
    catalogue_members = {}
    for p_id, genre, _ in texts:
        catalogue_members[p_id] = {"genre": genre, "project": "blms"}
    cat = {"members": catalogue_members}

    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("blms/catalogue.json", json.dumps(cat))
        for p_id, genre, cdl_words in texts:
            cdl = {"cdl": [{"node": "l", "f": {"lang": "sux", "form": w}}
                           for w in cdl_words]}
            zf.writestr(f"blms/corpusjson/{p_id}.json", json.dumps(cdl))
    return buf.getvalue()


def test_parse_incantation_zip_filters_by_genre():
    import io
    from pathlib import Path
    from unittest.mock import patch
    from languages.sumerian.scripts.fetch_incantations_11 import parse_incantation_zip, load_catalogue

    # One incantation text, one ritual text
    data = _make_blms_zip([
        ("P001", "Prayer/Incantation", ["an", "ki", "lugal"]),
        ("P002", "Ritual", ["ninda", "kasz"]),
    ])
    tmp_zip = io.BytesIO(data)

    # Write to a real temp path for ZipFile
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
        f.write(data)
        f.flush()
        tmp_path = Path(f.name)

    try:
        members = load_catalogue(tmp_path)
        docs = parse_incantation_zip(tmp_path, members)
        assert len(docs) == 1
        assert docs[0]["doc_id"] == "P001"
        assert docs[0]["raw_tokens"] == ["an", "ki", "lugal"]
    finally:
        os.unlink(tmp_path)


def test_parse_incantation_zip_no_catalogue_includes_all():
    import tempfile, os
    from pathlib import Path
    from languages.sumerian.scripts.fetch_incantations_11 import parse_incantation_zip

    # Build ZIP without catalogue.json
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        cdl = {"cdl": [{"node": "l", "f": {"lang": "sux", "form": "lugal"}}]}
        zf.writestr("blms/corpusjson/P001.json", json.dumps(cdl))
    data = buf.getvalue()

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
        f.write(data)
        tmp_path = Path(f.name)
    try:
        docs = parse_incantation_zip(tmp_path, {})
        assert len(docs) == 1
    finally:
        os.unlink(tmp_path)
