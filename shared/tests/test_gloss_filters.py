import pytest

from shared.scripts.gloss_filters import (
    DE_NEGATORS,
    MIN_HIT_RATE,
    NEGATORS,
    STOP_WORDS,
    check_hit_rate_gate,
    first_english,
    gw_is_usable,
    hit_rate_stats,
)

VOCAB = {"injuring", "harmlessness", "horns", "blade", "sea", "earring",
         "king", "compare", "cow"}


# --- first_english (dictionary-join slots) ---

def test_negated_gloss_rejected_entirely():
    # MW ahiṃsā: harvesting "injuring" would manufacture an antonym anchor
    assert first_english("not injuring anything", VOCAB) is None


def test_caller_falls_through_via_none():
    glosses = ["not injuring anything", "harmlessness"]
    picked = next((w for g in glosses if (w := first_english(g, VOCAB))), None)
    assert picked == "harmlessness"


def test_xref_gloss_rejected():
    assert first_english("see kṛṣṇa", VOCAB) is None
    assert first_english("cf the sea", VOCAB) is None


def test_xref_only_when_first_content_word():
    # "compare" is a genuine first gloss word here, not an xref marker
    assert first_english("compare the sea", VOCAB) == "compare"


def test_scaffold_words_skipped_not_harvested():
    assert first_english("having horns", VOCAB) == "horns"
    assert first_english("relating to the sea", VOCAB) == "sea"


def test_single_letter_skipped():
    assert first_english("c blade", VOCAB) == "blade"


def test_stop_words_still_skipped_and_not_no_are_negators():
    assert first_english("the sea", VOCAB) == "sea"
    assert "not" not in STOP_WORDS and "no" not in STOP_WORDS
    assert "not" in NEGATORS and "no" in NEGATORS


def test_hyphen_join_preserved():
    assert first_english("an ear-ring", VOCAB) == "earring"


def test_empty_and_all_scaffold():
    assert first_english("", VOCAB) is None
    assert first_english("having various", VOCAB) is None


# --- gw_is_usable (value-based slots) ---

def test_gw_plain_word_usable():
    assert gw_is_usable("king")
    assert gw_is_usable("to go")           # stop-word skipped, "go" is content


def test_gw_negation_led_rejected():
    assert not gw_is_usable("not injuring")
    assert not gw_is_usable("without form")


def test_gw_german_negators():
    assert not gw_is_usable("nicht verletzen", negators=DE_NEGATORS)
    assert not gw_is_usable("ohne Form", negators=DE_NEGATORS)
    assert gw_is_usable("König", negators=DE_NEGATORS)


def test_gw_xref_and_junk_rejected():
    assert not gw_is_usable("see previous")
    assert not gw_is_usable("c")
    assert not gw_is_usable("having")
    assert not gw_is_usable("")


# --- stats + gate ---

def test_hit_rate_stats_shape():
    s = hit_rate_stats(hits=3, misses=7, gloss_no_eng=1, anchors=2)
    assert s == {"hits": 3, "misses": 7, "token_hit_rate": 0.3,
                 "gloss_no_eng": 1, "anchors": 2}


def test_gate_fires_below_threshold():
    s = hit_rate_stats(hits=3, misses=7, gloss_no_eng=0, anchors=1)
    assert s["token_hit_rate"] < MIN_HIT_RATE
    with pytest.raises(SystemExit):
        check_hit_rate_gate(s, "MW")


def test_gate_passes_above_threshold():
    s = hit_rate_stats(hits=9, misses=1, gloss_no_eng=0, anchors=5)
    check_hit_rate_gate(s, "MW")  # must not raise
