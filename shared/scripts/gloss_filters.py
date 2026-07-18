"""
Canonical anchor-English gloss filters — suite v2.

Single source of truth for the noise classes measured in the 2026-07-14
survey and the Sanskrit slot build (journal 2026-07-16): negated glosses
harvested as antonym anchors, cross-reference glosses anchored to "see",
single-letter matches, and scaffold words harvested from gloss prose.

Two entry points, one per slot family:
  - first_english(): dictionary-join slots (greek/LSJ, sanskrit/MW) that scan
    gloss PROSE for the first usable in-vocab content word.
  - gw_is_usable(): value-based slots (sumerian ePSD2, akkadian, hittite,
    egyptian) whose English arrives as a short gw/english VALUE; they keep
    their own junk filters and vocab handling, and add these checks on top.
    Hittite passes DE_NEGATORS (its glosses are German; a negated German
    gloss embeds near its antonym through the translation step).

Also carries the anchor-stats payload and the 40% join-rate gate promoted
from the Sanskrit slot (PGM lesson) — the gate applies to dictionary-join
slots only; value slots have no equivalent join rate.
"""
from __future__ import annotations

import re

NEGATORS = frozenset({"not", "no", "without", "never"})
DE_NEGATORS = frozenset({"nicht", "kein", "keine", "keinen", "ohne", "nie",
                         "niemals"})
# A gloss whose FIRST content word is one of these is a cross-reference,
# not a meaning. ("q.v." never surfaces as a token under _WORD_RE; bare
# "q.v" segments are dropped by the 02 parsers' noise filters.)
XREF_STARTERS = frozenset({"see", "cf", "vid"})
# Gloss prose that is never a meaning. High-frequency verbs that are genuine
# glosses ("go", "act", "make", "kind") are deliberately NOT listed.
SCAFFOLD_WORDS = frozenset({
    "having", "relating", "belonging", "rarely", "who", "whose", "one's",
    "various", "especially", "particularly", "chiefly", "generally",
    "usually", "being",
})
# The Greek 06 set verbatim, minus "not"/"no" (those are NEGATORS: a negator
# must invalidate the gloss, not be skipped over).
STOP_WORDS = frozenset({
    "a", "an", "the", "to", "of", "in", "on", "at", "by", "for", "with",
    "be", "is", "are", "was", "were", "as", "or", "and",
    "but", "if", "so", "do", "did", "have", "has", "had", "from", "into",
    "out", "up", "down", "over", "under", "between", "during", "before",
    "after", "above", "below", "any", "some", "all", "each", "every",
    "one", "two", "three", "four", "five",
})
MIN_HIT_RATE = 0.40

# Unicode-aware: German umlaut glosses like "König" must tokenize as whole
# words (DE_NEGATORS path), not split at the umlaut.
_WORD_RE = re.compile(r"[^\W\d_](?:[^\W\d_]|['\-])*")


def first_english(gloss, eng_vocab_set, negators=NEGATORS):
    """First usable in-vocab content word of `gloss`, or None.

    Rules, in scan order over the gloss's words:
      - negator encountered before a match  -> None (whole gloss rejected)
      - stop/scaffold word                  -> skip, continue
      - first content word is an xref marker-> None (cross-reference gloss)
      - single-letter word                  -> skip, continue
      - word (or hyphen-joined form) in eng_vocab_set -> return it
    Callers fall through to the entry's next gloss on None.
    """
    if not gloss:
        return None
    seen_content = False
    for word in _WORD_RE.findall(gloss.lower()):
        if word in negators:
            return None
        if word in STOP_WORDS or word in SCAFFOLD_WORDS:
            continue
        if not seen_content and word in XREF_STARTERS:
            return None
        seen_content = True
        if len(word) == 1:
            continue
        if word in eng_vocab_set:
            return word
        if "-" in word:
            joined = word.replace("-", "")
            if joined in eng_vocab_set:
                return joined
    return None


def gw_is_usable(value, negators=NEGATORS):
    """Whether a short gw/english value can serve as anchor English.

    Rejects negation-led, xref-led, single-letter-first, and
    stop/scaffold-only values. No embedding-vocab check — that stays at fit
    time, as today. Verdict is based on the first content word.
    """
    if not value:
        return False
    for word in _WORD_RE.findall(value.lower()):
        if word in negators:
            return False
        if word in STOP_WORDS or word in SCAFFOLD_WORDS:
            continue
        if word in XREF_STARTERS:
            return False
        return len(word) > 1
    return False


def hit_rate_stats(hits, misses, gloss_no_eng, anchors):
    """Canonical anchor_stats.json payload for dictionary-join slots."""
    return {
        "hits": hits,
        "misses": misses,
        "token_hit_rate": hits / max(1, hits + misses),
        "gloss_no_eng": gloss_no_eng,
        "anchors": anchors,
    }


def check_hit_rate_gate(stats, source_name):
    """SystemExit if the join hit rate is below MIN_HIT_RATE. Call AFTER
    persisting anchors + stats so the evidence survives the stop."""
    if stats["token_hit_rate"] < MIN_HIT_RATE:
        raise SystemExit(
            f"{source_name} join hit rate {stats['token_hit_rate']:.1%} is "
            f"below the {MIN_HIT_RATE:.0%} gate (PGM lesson). Inspect lemma "
            "normalization / lexicon parse before any FastText compute."
        )
