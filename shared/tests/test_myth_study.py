import numpy as np
import pytest

from shared.scripts.doc_eval import ETCSL_PATH
from shared.scripts.myth_study import (
    THEMES,
    HITTITE_MERGES,
    build_roster,
    load_corpora,
)


# ---------- roster invariants (real corpora) ----------

needs_data = pytest.mark.skipif(not ETCSL_PATH.exists(),
                                reason="corpora not available in this checkout")


@pytest.fixture(scope="module")
def roster_bundle():
    corpora = load_corpora()
    roster, notes, roster_tokens = build_roster(corpora)
    return corpora, roster, notes, roster_tokens


@needs_data
def test_roster_theme_keys_and_min_viability(roster_bundle):
    _, roster, _, _ = roster_bundle
    for slot in ("sumerian", "hittite", "greek"):
        assert set(roster[slot]) == set(THEMES)
        for theme in ("cosmogonic", "hymnic", "royal_control"):
            assert roster[slot][theme], f"{slot}/{theme} unexpectedly empty"


@needs_data
def test_roster_no_duplicate_doc_ids(roster_bundle):
    _, roster, _, _ = roster_bundle
    for slot in roster:
        ids = [e["doc_id"] for theme in THEMES for e in roster[slot][theme]]
        assert len(ids) == len(set(ids)), f"duplicate doc ids in {slot}"


@needs_data
def test_roster_pinned_docs_present(roster_bundle):
    _, roster, _, _ = roster_bundle
    sum_cosmo = {e["doc_id"] for e in roster["sumerian"]["cosmogonic"]}
    assert sum_cosmo == {"c141", "c174", "c111", "c112", "c113"}
    hit_cosmo = {e["doc_id"] for e in roster["hittite"]["cosmogonic"]}
    assert hit_cosmo == {"kumarbi", "ullikummi", "illuyanka"}
    gre_cosmo = {e["doc_id"] for e in roster["greek"]["cosmogonic"]}
    assert gre_cosmo == {"Hesiod (0020) - Theogony (001)",
                         "Hesiod (0020) - Works and Days (002)"}


@needs_data
def test_hittite_merged_union_token_counts(roster_bundle):
    corpora, roster, _, roster_tokens = roster_bundle
    hdocs = corpora["docs"]["hittite"]
    by_id = {e["doc_id"]: e for e in roster["hittite"]["cosmogonic"]}
    for merged_id, components in HITTITE_MERGES.items():
        expected = sum(len(hdocs[c]) for c in components)
        assert by_id[merged_id]["n_tokens"] == expected
        assert len(roster_tokens["hittite"][merged_id]) == expected
        assert list(by_id[merged_id]["components"]) == list(components)


@needs_data
def test_empty_themes_logged_with_reasons(roster_bundle):
    _, roster, notes, _ = roster_bundle
    assert roster["sumerian"]["magical"] == []
    assert notes["sumerian"]["magical"]["reason_empty"]
    assert roster["greek"]["magical"] == []
    assert "PGM" in notes["greek"]["magical"]["reason_empty"]
    assert roster["greek"]["wisdom"] == []
    assert notes["greek"]["wisdom"]["reason_empty"]
