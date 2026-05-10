"""
Sumerian Anchor Coverage Diagnostic (Workstream 2b-pre).

Takes the akkadian_vocab_miss anchors from the Workstream 2a audit and:
  1. Classifier: attributes each to ONE primary cause (priority-ordered).
  2. Simulator: reports per-intervention projected recovery, with Tier-2
     semantic validation for the two inference-based interventions.

Writes a dated report pair under `results/`:
  - coverage_diagnostic_<YYYY-MM-DD>.md
  - coverage_diagnostic_<YYYY-MM-DD>.json  (schema version 1)

See: docs/superpowers/specs/2026-04-19-coverage-diagnostic-design.md
"""
from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure repo root is importable when invoked directly (pytest.ini only affects pytest).
_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

# akkadian_normalize is always available; audit_anchors is imported lazily inside
# run_diagnostic so the module can be loaded for unit-testing without it.
from languages.akkadian.scripts.akkadian_normalize import normalize_akkadian_token

# --- Module constants -------------------------------------------------------

DIAGNOSTIC_SCHEMA_VERSION = 1
DEFAULT_SEED = 42
SUBWORD_OVERLAP_MIN = 0.5


# --- DiagnosticContext ------------------------------------------------------


@dataclass(frozen=True)
class DiagnosticContext:
    fused_vocab: frozenset[str]
    glove_vocab: frozenset[str]
    gemma_vocab: frozenset[str]
    corpus_frequency: dict[str, int]
    lemma_surface_map: dict[str, frozenset[str]]
    fasttext_model: Any  # gensim FastText model, or None for tests that don't need it
    gemma_english_vocab: list[str]
    gemma_english_vectors: np.ndarray  # (N, 768) float32
    ridge_gemma_coef: np.ndarray       # (768, 1536) float32
    ridge_gemma_intercept: np.ndarray  # (768,) float32


# --- Loaders ----------------------------------------------------------------


def _load_corpus_frequency(path: Path) -> dict[str, int]:
    """Count token occurrences in the cleaned corpus (whitespace-split per line)."""
    freq: Counter[str] = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            for token in line.strip().split():
                freq[token] += 1
    return dict(freq)


def _load_lemma_surface_map(path: Path) -> dict[str, frozenset[str]]:
    """Build {citation_form: {surface_forms}} from ORACC lemmas.

    Applies the same ORACC->ATF normalization used in scripts/06_extract_anchors.py
    so the surface forms match what the corpus (and thus FastText vocab) contains.
    Empty cf or form values are skipped.
    """
    import json
    with open(path, encoding="utf-8") as f:
        lemmas = json.load(f)

    surfaces_by_cf: dict[str, set[str]] = {}
    for lemma in lemmas:
        cf_raw = (lemma.get("cf") or "").strip()
        form_raw = (lemma.get("form") or "").strip()
        if not cf_raw or not form_raw:
            continue
        cf = normalize_akkadian_token(cf_raw)
        form = normalize_akkadian_token(form_raw)
        if not cf or not form:
            continue
        surfaces_by_cf.setdefault(cf, set()).add(form)

    # Freeze the surface sets to match the frozen DiagnosticContext contract.
    return {cf: frozenset(forms) for cf, forms in surfaces_by_cf.items()}


def _load_fasttext_model(path: Path):
    """Load the full FastText model (needed for subword n-gram access)."""
    from gensim.models import FastText
    return FastText.load(str(path))


def _load_ridge_weights(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ridge coef + intercept from an npz produced by align_and_evaluate."""
    data = np.load(str(path))
    coef = data["coef"].astype(np.float32)
    intercept = data["intercept"].astype(np.float32)
    if coef.shape[0] != 768 or coef.shape[1] != 1536:
        raise ValueError(
            f"Ridge coef shape {coef.shape} != (768, 1536) — "
            "expected whitened-Gemma ridge weights"
        )
    if intercept.shape != (768,):
        raise ValueError(
            f"Ridge intercept shape {intercept.shape} != (768,)"
        )
    return coef, intercept


def _load_gemma_english_npz(path: Path) -> tuple[list[str], np.ndarray]:
    """Load the whitened-Gemma English cache (vocab + vectors)."""
    data = np.load(str(path))
    vocab = [str(w) for w in data["vocab"]]
    vectors = data["vectors"].astype(np.float32)
    if vectors.shape[1] != 768:
        raise ValueError(
            f"Gemma vectors dim {vectors.shape[1]} != 768 — "
            "regenerate via scripts/whiten_gemma.py"
        )
    if vectors.shape[0] != len(vocab):
        raise ValueError("Gemma vocab/vectors row count mismatch")
    return vocab, vectors



# --- n-gram helpers --------------------------------------------------------


def _ngrams(word: str, min_n: int, max_n: int) -> frozenset[str]:
    """Character n-grams with FastText-style angle-bracket padding."""
    padded = f"<{word}>"
    result: set[str] = set()
    for n in range(min_n, max_n + 1):
        if n > len(padded):
            continue
        for i in range(len(padded) - n + 1):
            result.add(padded[i : i + n])
    return frozenset(result)


def _trained_ngrams(vocab, min_n: int, max_n: int) -> frozenset[str]:
    """Union of n-grams across the training vocab. Expensive once, fast per lookup."""
    out: set[str] = set()
    for word in vocab:
        out.update(_ngrams(word, min_n, max_n))
    return frozenset(out)


def _subword_overlap(anchor: str, trained_ngrams: frozenset[str], min_n: int, max_n: int) -> float:
    anchor_ngrams = _ngrams(anchor, min_n, max_n)
    if not anchor_ngrams:
        return 0.0
    return len(anchor_ngrams & trained_ngrams) / len(anchor_ngrams)


# --- Classifier ------------------------------------------------------------

PRIMARY_CAUSE_ORDER = (
    "normalization_recoverable",
    "in_corpus_below_min_count",
    "oracc_lemma_surface_recoverable",
    "morpheme_composition_recoverable",
    "subword_inference_recoverable",
    "logogram_unmatched",
    "genuinely_missing",
)


def _morphemes(anchor_raw: str) -> list[str]:
    """Split by hyphens, normalize each morpheme individually."""
    if "-" not in anchor_raw:
        return []
    parts: list[str] = []
    for piece in anchor_raw.split("-"):
        # normalize_akkadian_token handles subscripts, braces, ORACC->ATF, and lowercase.
        # Hyphens are already split out, so the hyphen-drop step is a no-op per piece.
        piece = normalize_akkadian_token(piece)
        if piece:
            parts.append(piece)
    return parts


def classify_miss(
    anchor: dict,
    ctx: "DiagnosticContext",
    trained_ngrams: frozenset[str],
    fasttext_min_n: int = 3,
    fasttext_max_n: int = 6,
) -> str:
    """Priority-ordered primary-cause attribution for one missing anchor."""
    akkadian_raw = str(anchor.get("akkadian") or "").strip()
    english = str(anchor.get("english") or "").lower()

    # 1. normalization_recoverable
    normalized = normalize_akkadian_token(akkadian_raw)
    if normalized and normalized in ctx.fused_vocab:
        return "normalization_recoverable"

    # 2. in_corpus_below_min_count: check both raw and normalized forms.
    for candidate in (akkadian_raw, normalized):
        if not candidate:
            continue
        count = ctx.corpus_frequency.get(candidate, 0)
        if 1 <= count < 5:
            return "in_corpus_below_min_count"

    # 3. oracc_lemma_surface_recoverable: citation form (normalized) -> any surface in vocab.
    if normalized in ctx.lemma_surface_map:
        for surface in ctx.lemma_surface_map[normalized]:
            if surface in ctx.fused_vocab:
                return "oracc_lemma_surface_recoverable"

    # 4. morpheme_composition_recoverable: hyphenated, all morphemes in vocab.
    morphemes = _morphemes(akkadian_raw)
    if morphemes and all(m in ctx.fused_vocab for m in morphemes):
        return "morpheme_composition_recoverable"

    # 5. subword_inference_recoverable: n-gram overlap >= threshold on NORMALIZED form.
    if normalized:
        overlap = _subword_overlap(normalized, trained_ngrams, fasttext_min_n, fasttext_max_n)
        if overlap >= SUBWORD_OVERLAP_MIN:
            return "subword_inference_recoverable"

    # 6. logogram_unmatched: anchor surface is fully uppercase (a Sumerogram
    # in Akkadian text). The corpus tokenizer drops ALL-CAPS signs in
    # clean_atf_line, so all-uppercase anchors miss systematically.
    if akkadian_raw and akkadian_raw.isupper():
        return "logogram_unmatched"

    return "genuinely_missing"


def classify_all_misses(
    misses: list[dict],
    ctx: "DiagnosticContext",
    trained_ngrams: frozenset[str],
    fasttext_min_n: int = 3,
    fasttext_max_n: int = 6,
) -> dict:
    """Classify every miss; return totals + per-bucket rows with traces."""
    primary_causes: dict[str, list[dict]] = {name: [] for name in PRIMARY_CAUSE_ORDER}

    for anchor in misses:
        bucket = classify_miss(anchor, ctx, trained_ngrams, fasttext_min_n, fasttext_max_n)
        # Attach a lightweight trace describing the match, for the examples field.
        trace: dict[str, Any] = {}
        akkadian_raw = str(anchor.get("akkadian") or "").strip()
        normalized = normalize_akkadian_token(akkadian_raw)
        if bucket == "normalization_recoverable":
            trace["normalized_form"] = normalized
        elif bucket == "in_corpus_below_min_count":
            for candidate in (akkadian_raw, normalized):
                if 1 <= ctx.corpus_frequency.get(candidate, 0) < 5:
                    trace["matched_form"] = candidate
                    trace["corpus_count"] = ctx.corpus_frequency[candidate]
                    break
        elif bucket == "oracc_lemma_surface_recoverable":
            hits = [s for s in ctx.lemma_surface_map.get(normalized, ()) if s in ctx.fused_vocab]
            trace["matched_surface_forms"] = hits[:3]
        elif bucket == "morpheme_composition_recoverable":
            trace["morphemes_in_vocab"] = _morphemes(akkadian_raw)
        elif bucket == "subword_inference_recoverable":
            overlap = _subword_overlap(normalized, trained_ngrams, fasttext_min_n, fasttext_max_n)
            trace["ngram_overlap"] = round(overlap, 4)
        elif bucket == "logogram_unmatched":
            trace["uppercase_form"] = akkadian_raw

        enriched = dict(anchor)
        enriched["trace"] = trace
        primary_causes[bucket].append(enriched)

    total = len(misses)
    return {
        "total_misses": total,
        "primary_causes": {
            name: {
                "count": len(rows),
                "pct": (len(rows) / total * 100.0) if total else 0.0,
                "rows": rows,
            }
            for name, rows in primary_causes.items()
        },
    }


# --- Simulators (exact-match) ----------------------------------------------


def simulate_ascii_normalize(misses: list[dict], ctx: "DiagnosticContext") -> dict:
    """Apply normalization; count anchors whose normalized form is in explicit vocab."""
    resolvable = 0
    for anchor in misses:
        normalized = normalize_akkadian_token(str(anchor.get("akkadian") or ""))
        if normalized and normalized in ctx.fused_vocab:
            resolvable += 1
    return {
        "anchors_newly_resolvable": resolvable,
        "trustworthiness": "exact",
        "notes": "Pure string normalization; recovered anchors are exact vocab matches.",
    }


def simulate_lower_min_count(misses: list[dict], ctx: "DiagnosticContext") -> dict:
    """For each min_count in {1,2,3,4}, count OOV anchors with corpus freq >= that threshold."""
    per_threshold: dict[str, dict] = {}
    for threshold in (1, 2, 3, 4):
        resolvable = 0
        for anchor in misses:
            akkadian_raw = str(anchor.get("akkadian") or "").strip()
            normalized = normalize_akkadian_token(akkadian_raw)
            # Skip if the form is ALREADY in explicit vocab (not a miss).
            if akkadian_raw in ctx.fused_vocab or normalized in ctx.fused_vocab:
                continue
            for candidate in (akkadian_raw, normalized):
                if not candidate:
                    continue
                if ctx.corpus_frequency.get(candidate, 0) >= threshold:
                    resolvable += 1
                    break
        per_threshold[str(threshold)] = {
            "anchors_newly_resolvable": resolvable,
        }
    return {
        "per_threshold": per_threshold,
        "trustworthiness": "exact",
        "notes": "Assumes FastText retrain; anchor form exists in cleaned_corpus.txt at the stated frequency.",
    }


def simulate_oracc_lemma_expansion(misses: list[dict], ctx: "DiagnosticContext") -> dict:
    """Count anchors whose citation form has any surface variant in explicit vocab."""
    resolvable = 0
    unique_surfaces: set[str] = set()
    for anchor in misses:
        normalized = normalize_akkadian_token(str(anchor.get("akkadian") or ""))
        surfaces = ctx.lemma_surface_map.get(normalized, frozenset())
        matched = [s for s in surfaces if s in ctx.fused_vocab]
        if matched:
            resolvable += 1
            unique_surfaces.update(matched)
    return {
        "anchors_newly_resolvable": resolvable,
        "surface_forms_added_to_vocab": len(unique_surfaces),
        "trustworthiness": "exact",
        "notes": "Expansion maps each citation form to every surface variant in FastText vocab.",
    }


# --- Tier-2 semantic validation --------------------------------------------


def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def _l2_normalize_vec(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm == 0:
        return v
    return v / norm


def _project_ft_to_gemma(ft_vec: np.ndarray, ctx: "DiagnosticContext") -> np.ndarray:
    """Fuse with zero-padding, project through Gemma ridge."""
    ft_vec = ft_vec.astype(np.float32)
    fused = np.concatenate([ft_vec, np.zeros(768, dtype=np.float32)])  # (1536,)
    projected = fused @ ctx.ridge_gemma_coef.T + ctx.ridge_gemma_intercept  # (768,)
    return projected


def _tier2_nearest_english(ft_vec: np.ndarray, ctx: "DiagnosticContext", k: int) -> list[str]:
    projected = _project_ft_to_gemma(ft_vec, ctx)
    query = _l2_normalize_vec(projected)
    eng_norm = _l2_normalize_rows(ctx.gemma_english_vectors)
    sims = eng_norm @ query  # (N,)
    top_idx = np.argsort(sims)[::-1][:k]
    return [ctx.gemma_english_vocab[int(i)] for i in top_idx]


def _tier2_score_anchor(
    ft_vec: np.ndarray,
    expected_english: str,
    ctx: "DiagnosticContext",
) -> dict:
    top10 = _tier2_nearest_english(ft_vec, ctx, k=10)
    expected = expected_english.lower()
    return {
        "top1": len(top10) >= 1 and top10[0] == expected,
        "top5": expected in top10[:5],
        "top10": expected in top10[:10],
    }


# --- Inference simulators --------------------------------------------------


def simulate_morpheme_composition(
    misses: list[dict],
    ctx: "DiagnosticContext",
    *,
    morpheme_vector_lookup,
) -> dict:
    """Simulator #4: hyphenated anchors; all morphemes in vocab; vector = mean."""
    resolvable_tier1 = 0
    tier2_tested = 0
    tier2_top1 = 0
    tier2_top5 = 0
    tier2_top10 = 0
    tier2_skipped = 0

    for anchor in misses:
        akkadian_raw = str(anchor.get("akkadian") or "").strip()
        morphemes = _morphemes(akkadian_raw)
        if not morphemes:
            continue
        if not all(m in ctx.fused_vocab for m in morphemes):
            continue

        resolvable_tier1 += 1

        # Tier-2: check if expected English is in Gemma vocab.
        expected = str(anchor.get("english") or "").lower()
        if expected not in ctx.gemma_vocab:
            tier2_skipped += 1
            continue

        # Synthesize morpheme-mean vector.
        vecs = []
        for m in morphemes:
            v = morpheme_vector_lookup(m)
            if v is None:
                break
            vecs.append(np.asarray(v, dtype=np.float32))
        if len(vecs) != len(morphemes):
            # Lookup missed a morpheme (shouldn't happen per vocab check, but guard).
            tier2_skipped += 1
            continue
        synthesized = np.mean(np.stack(vecs, axis=0), axis=0)

        score = _tier2_score_anchor(synthesized, expected, ctx)
        tier2_tested += 1
        if score["top1"]: tier2_top1 += 1
        if score["top5"]: tier2_top5 += 1
        if score["top10"]: tier2_top10 += 1

    return {
        "anchors_newly_resolvable_tier1": resolvable_tier1,
        "tier2_semantic": {
            "tested": tier2_tested,
            "top1_correct": tier2_top1,
            "top5_correct": tier2_top5,
            "top10_correct": tier2_top10,
            "skipped": tier2_skipped,
        },
        "trustworthiness": "inferred (compositional)",
        "notes": "Vector = numpy mean of constituent morpheme vectors. Tier-2 checks whitened-Gemma projection.",
    }


def simulate_subword_inference(
    misses: list[dict],
    ctx: "DiagnosticContext",
    *,
    trained_ngrams: frozenset[str],
    subword_vector_lookup,
    fasttext_min_n: int = 3,
    fasttext_max_n: int = 6,
) -> dict:
    """Simulator #5: FastText OOV inference with >= SUBWORD_OVERLAP_MIN n-gram overlap."""
    resolvable_tier1 = 0
    tier2_tested = 0
    tier2_top1 = 0
    tier2_top5 = 0
    tier2_top10 = 0
    tier2_skipped = 0

    for anchor in misses:
        akkadian_raw = str(anchor.get("akkadian") or "").strip()
        normalized = normalize_akkadian_token(akkadian_raw)
        if not normalized:
            continue
        # Skip anchors already in vocab (not misses) — defensive.
        if normalized in ctx.fused_vocab:
            continue
        overlap = _subword_overlap(normalized, trained_ngrams, fasttext_min_n, fasttext_max_n)
        if overlap < SUBWORD_OVERLAP_MIN:
            continue

        resolvable_tier1 += 1

        expected = str(anchor.get("english") or "").lower()
        if expected not in ctx.gemma_vocab:
            tier2_skipped += 1
            continue

        ft_vec = subword_vector_lookup(normalized)
        if ft_vec is None:
            tier2_skipped += 1
            continue
        ft_vec = np.asarray(ft_vec, dtype=np.float32)

        score = _tier2_score_anchor(ft_vec, expected, ctx)
        tier2_tested += 1
        if score["top1"]: tier2_top1 += 1
        if score["top5"]: tier2_top5 += 1
        if score["top10"]: tier2_top10 += 1

    return {
        "anchors_newly_resolvable_tier1": resolvable_tier1,
        "tier2_semantic": {
            "tested": tier2_tested,
            "top1_correct": tier2_top1,
            "top5_correct": tier2_top5,
            "top10_correct": tier2_top10,
            "skipped": tier2_skipped,
        },
        "trustworthiness": "inferred (character n-gram)",
        "notes": "Uses FastText.wv.get_vector for OOV. Tier-2 checks whitened-Gemma projection.",
    }


# --- Rendering -------------------------------------------------------------


RECOVERABILITY_TAGS = {
    "normalization_recoverable":        "exact — normalize and rerun extraction",
    "in_corpus_below_min_count":        "exact — lower min_count and retrain",
    "oracc_lemma_surface_recoverable":  "exact — expand anchors to ORACC surface forms",
    "morpheme_composition_recoverable": "inferred — compose vector from morphemes in vocab",
    "subword_inference_recoverable":    "inferred — use FastText OOV subword inference",
    "logogram_unmatched":               "low — ALL-CAPS Sumerograms dropped by corpus tokenizer",
    "genuinely_missing":                "none — requires new corpus or lemma data",
}


def _escape_md_cell(s) -> str:
    return str(s).replace("|", r"\|")


def _pick_examples(rows: list[dict], n: int, seed: int) -> list[dict]:
    if not rows:
        return []
    n = min(n, len(rows))
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(rows), size=n, replace=False).tolist())
    return [rows[i] for i in indices]


def render_json(
    classifier_result: dict,
    simulator_result: dict,
    metadata: dict,
    examples_per_bucket: int = 10,
) -> dict:
    seed = metadata.get("source_artifacts", {}).get("seed", DEFAULT_SEED)
    primary_causes_out = {}
    for name in PRIMARY_CAUSE_ORDER:
        bucket = classifier_result["primary_causes"][name]
        primary_causes_out[name] = {
            "count": bucket["count"],
            "pct": round(bucket["pct"], 4),
            "examples": _pick_examples(bucket.get("rows", []), examples_per_bucket, seed),
        }
    return {
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_date": metadata["diagnostic_date"],
        "source_artifacts": metadata["source_artifacts"],
        "baseline": metadata["baseline"],
        "classifier": {
            "total_misses": classifier_result["total_misses"],
            "primary_causes": primary_causes_out,
        },
        "simulator": simulator_result,
    }


def _format_anchor_row(row: dict) -> str:
    trace = row.get("trace") or {}
    trace_str = ", ".join(f"{k}={v!r}" for k, v in trace.items())
    return (
        f"| {_escape_md_cell(row.get('sumerian', ''))} "
        f"| {_escape_md_cell(row.get('english', ''))} "
        f"| {row.get('confidence', 0):.3f} "
        f"| {_escape_md_cell(trace_str)} |"
    )


def _ranked_interventions(simulator_result: dict) -> list[tuple[str, int, str]]:
    """Return [(name, projected_count, trustworthiness_tag), ...] sorted desc by count.

    For lower_min_count uses t=1 (most permissive); for inference-based uses Tier-2 top-5.
    """
    interventions = simulator_result["interventions"]
    rows = []
    rows.append(("ascii_normalize",
                 interventions["ascii_normalize"]["anchors_newly_resolvable"],
                 interventions["ascii_normalize"]["trustworthiness"]))
    rows.append(("lower_min_count (t=1)",
                 interventions["lower_min_count"]["per_threshold"]["1"]["anchors_newly_resolvable"],
                 interventions["lower_min_count"]["trustworthiness"]))
    rows.append(("oracc_lemma_expansion",
                 interventions["oracc_lemma_expansion"]["anchors_newly_resolvable"],
                 interventions["oracc_lemma_expansion"]["trustworthiness"]))
    rows.append(("morpheme_composition (Tier-2 top-5)",
                 interventions["morpheme_composition"]["tier2_semantic"]["top5_correct"],
                 interventions["morpheme_composition"]["trustworthiness"]))
    rows.append(("subword_inference (Tier-2 top-5)",
                 interventions["subword_inference"]["tier2_semantic"]["top5_correct"],
                 interventions["subword_inference"]["trustworthiness"]))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def render_markdown(
    classifier_result: dict,
    simulator_result: dict,
    metadata: dict,
    examples_per_bucket: int = 10,
) -> str:
    baseline = metadata["baseline"]
    seed = metadata.get("source_artifacts", {}).get("seed", DEFAULT_SEED)
    total_misses = classifier_result["total_misses"]

    lines: list[str] = []
    lines.append(f"# Coverage Diagnostic — {metadata['diagnostic_date']}")
    lines.append("")
    lines.append("Generated by `scripts/coverage_diagnostic.py`. See design spec at "
                 "`docs/superpowers/specs/2026-04-19-coverage-diagnostic-design.md`.")
    lines.append("")

    lines.append("## Baseline")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Total merged anchors | {baseline['total_merged']:,} |")
    lines.append(f"| Surviving (both target vocabs) | {baseline['survives']:,} |")
    lines.append(f"| akkadian_vocab_miss (this diagnostic's input) | {baseline['akkadian_vocab_miss']:,} |")
    lines.append("")

    lines.append("## Classifier — primary-cause attribution (mutually exclusive)")
    lines.append("")
    lines.append("| Primary cause | Count | % of misses | Trustworthiness |")
    lines.append("|---|---:|---:|---|")
    for name in PRIMARY_CAUSE_ORDER:
        bucket = classifier_result["primary_causes"][name]
        lines.append(f"| `{name}` | {bucket['count']:,} | {bucket['pct']:.2f}% | {RECOVERABILITY_TAGS[name]} |")
    lines.append("")

    lines.append("## Classifier — example rows per bucket")
    lines.append("")
    lines.append(f"Up to {examples_per_bucket} deterministically-sampled rows per non-empty bucket.")
    lines.append("")
    for name in PRIMARY_CAUSE_ORDER:
        bucket = classifier_result["primary_causes"][name]
        rows = bucket.get("rows", [])
        examples = _pick_examples(rows, examples_per_bucket, seed)
        if not examples:
            continue
        lines.append(f"### `{name}` ({bucket['count']:,} rows)")
        lines.append("")
        lines.append("| sumerian | english | confidence | trace |")
        lines.append("|---|---|---:|---|")
        for row in examples:
            lines.append(_format_anchor_row(row))
        lines.append("")

    lines.append("## Simulator — per-intervention projected recovery")
    lines.append("")
    sim = simulator_result["interventions"]
    lines.append("### `ascii_normalize`")
    lines.append(f"- anchors_newly_resolvable: **{sim['ascii_normalize']['anchors_newly_resolvable']:,}**")
    lines.append(f"- trustworthiness: {sim['ascii_normalize']['trustworthiness']}")
    lines.append("")
    lines.append("### `lower_min_count`")
    lines.append("")
    lines.append("| min_count | anchors_newly_resolvable |")
    lines.append("|---:|---:|")
    for t in ("1", "2", "3", "4"):
        lines.append(f"| {t} | {sim['lower_min_count']['per_threshold'][t]['anchors_newly_resolvable']:,} |")
    lines.append(f"- trustworthiness: {sim['lower_min_count']['trustworthiness']}")
    lines.append("")
    lines.append("### `oracc_lemma_expansion`")
    lines.append(f"- anchors_newly_resolvable: **{sim['oracc_lemma_expansion']['anchors_newly_resolvable']:,}**")
    lines.append(f"- surface_forms_added_to_vocab: {sim['oracc_lemma_expansion']['surface_forms_added_to_vocab']:,}")
    lines.append(f"- trustworthiness: {sim['oracc_lemma_expansion']['trustworthiness']}")
    lines.append("")
    for name in ("morpheme_composition", "subword_inference"):
        inf = sim[name]
        t2 = inf["tier2_semantic"]
        lines.append(f"### `{name}`")
        lines.append(f"- anchors_newly_resolvable_tier1: **{inf['anchors_newly_resolvable_tier1']:,}**")
        lines.append(f"- Tier-2: tested={t2['tested']:,}, top1_correct={t2['top1_correct']:,}, "
                     f"top5_correct={t2['top5_correct']:,}, top10_correct={t2['top10_correct']:,}, "
                     f"skipped={t2.get('skipped', 0):,}")
        lines.append(f"- trustworthiness: {inf['trustworthiness']}")
        lines.append("")

    lines.append("## Ranked intervention recommendations")
    lines.append("")
    lines.append("| Intervention | Projected recoverable | Trustworthiness |")
    lines.append("|---|---:|---|")
    for name, count, tag in _ranked_interventions(simulator_result):
        lines.append(f"| {name} | {count:,} | {tag} |")
    lines.append("")

    lines.append("## Methodology notes")
    lines.append("")
    lines.append("- **Exact** trustworthiness = the intervention produces a vocab hit (no inference). "
                 "**Inferred** = the intervention synthesizes a vector from subwords/morphemes.")
    lines.append("- Tier-2 semantic validation projects synthesized Sumerian vectors through "
                 "`models/ridge_weights_gemma_whitened.npz` and checks whether the expected "
                 "English gloss is among the top-K Gemma nearest neighbors.")
    lines.append("- Interventions in the simulator table are INDEPENDENT — their counts can overlap. "
                 "The classifier table gives mutually-exclusive primary-cause attribution.")
    lines.append("- All input artifacts are SHA-256 stamped in the JSON report for future diff-runs.")
    lines.append("")

    return "\n".join(lines)


# --- Main ------------------------------------------------------------------

import argparse  # noqa: E402
import datetime as _dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def run_diagnostic(
    *,
    anchors_path: Path,
    fused_path: Path,
    glove_path: Path,
    gemma_path: Path,
    ridge_gemma_path: Path,
    oracc_lemmas_path: Path,
    cleaned_corpus_path: Path,
    fasttext_model_path: Path,
    out_dir: Path,
    diagnostic_date: str,
    examples_per_bucket: int = 10,
    seed: int = DEFAULT_SEED,
) -> int:
    # Hard require all inputs.
    for p in (anchors_path, fused_path, glove_path, gemma_path, ridge_gemma_path,
              oracc_lemmas_path, cleaned_corpus_path, fasttext_model_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Required input missing: {p}. "
                "See docs/superpowers/specs/2026-04-19-coverage-diagnostic-design.md for generators."
            )

    # Lazy import: audit_anchors may not exist during unit-test runs.
    from languages.akkadian.scripts.audit_anchors import (  # noqa: PLC0415
        _load_anchors,
        _load_fused_vocab,
        _load_gemma_vocab,
        _load_glove_vocab,
    )

    # Load everything.
    anchors = _load_anchors(anchors_path)
    fused_vocab = frozenset(_load_fused_vocab(fused_path))
    glove_vocab = frozenset(_load_glove_vocab(glove_path))
    gemma_vocab = frozenset(_load_gemma_vocab(gemma_path))
    gemma_english_vocab, gemma_english_vectors = _load_gemma_english_npz(gemma_path)
    ridge_coef, ridge_intercept = _load_ridge_weights(ridge_gemma_path)
    corpus_freq = _load_corpus_frequency(cleaned_corpus_path)
    lemma_map = _load_lemma_surface_map(oracc_lemmas_path)
    ft_model = _load_fasttext_model(fasttext_model_path)

    # Use the audit's classifier to find misses against BOTH target vocabs.
    # The akkadian_vocab_miss bucket is our input set.
    from languages.akkadian.scripts.audit_anchors import AuditContext, classify_all
    audit_ctx = AuditContext(
        fused_vocab=fused_vocab,
        glove_vocab=glove_vocab,
        gemma_vocab=gemma_vocab,
        collision_keys=frozenset(),  # irrelevant here; we only need the vocab_miss bucket
    )
    audit_result = classify_all(anchors, audit_ctx)
    misses = audit_result["buckets"]["akkadian_vocab_miss"]["rows"]
    baseline = {
        "total_merged": audit_result["totals"]["merged"],
        "survives": audit_result["totals"]["survives"],
        "akkadian_vocab_miss": audit_result["buckets"]["akkadian_vocab_miss"]["count"],
    }

    # Build the DiagnosticContext.
    ctx = DiagnosticContext(
        fused_vocab=fused_vocab,
        glove_vocab=glove_vocab,
        gemma_vocab=gemma_vocab,
        corpus_frequency=corpus_freq,
        lemma_surface_map=lemma_map,
        fasttext_model=ft_model,
        gemma_english_vocab=gemma_english_vocab,
        gemma_english_vectors=gemma_english_vectors,
        ridge_gemma_coef=ridge_coef,
        ridge_gemma_intercept=ridge_intercept,
    )

    # Trained n-grams (computed once, used by classifier + subword simulator).
    min_n = int(ft_model.wv.min_n)
    max_n = int(ft_model.wv.max_n)
    trained_ngrams = _trained_ngrams(fused_vocab, min_n, max_n)

    # Classifier.
    classifier_result = classify_all_misses(misses, ctx, trained_ngrams, min_n, max_n)

    # Morpheme lookup: use explicit vocab vectors from the FastText model.
    def morpheme_lookup(morpheme: str):
        if morpheme in ft_model.wv:
            return ft_model.wv[morpheme]
        return None

    # Subword lookup: FastText OOV inference via wv.get_vector.
    def subword_lookup(word: str):
        try:
            return ft_model.wv.get_vector(word)
        except KeyError:
            return None

    # Simulators.
    sim_ascii = simulate_ascii_normalize(misses, ctx)
    sim_lower = simulate_lower_min_count(misses, ctx)
    sim_lemma = simulate_oracc_lemma_expansion(misses, ctx)
    sim_morph = simulate_morpheme_composition(misses, ctx, morpheme_vector_lookup=morpheme_lookup)
    sim_sub = simulate_subword_inference(
        misses, ctx,
        trained_ngrams=trained_ngrams,
        subword_vector_lookup=subword_lookup,
        fasttext_min_n=min_n, fasttext_max_n=max_n,
    )
    simulator_result = {
        "interventions": {
            "ascii_normalize": sim_ascii,
            "lower_min_count": sim_lower,
            "oracc_lemma_expansion": sim_lemma,
            "morpheme_composition": sim_morph,
            "subword_inference": sim_sub,
        },
    }

    metadata = {
        "diagnostic_date": diagnostic_date,
        "source_artifacts": {
            "anchors_path": str(anchors_path),
            "anchors_sha256": _sha256(anchors_path),
            "fasttext_model_path": str(fasttext_model_path),
            "fasttext_model_sha256": _sha256(fasttext_model_path),
            "fused_vocab_path": str(fused_path),
            "glove_path": str(glove_path),
            "gemma_path": str(gemma_path),
            "ridge_gemma_path": str(ridge_gemma_path),
            "oracc_lemmas_path": str(oracc_lemmas_path),
            "cleaned_corpus_path": str(cleaned_corpus_path),
            "cleaned_corpus_sha256": _sha256(cleaned_corpus_path),
            "seed": seed,
            "subword_overlap_min": SUBWORD_OVERLAP_MIN,
        },
        "baseline": baseline,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"coverage_diagnostic_{diagnostic_date}.json"
    md_path = out_dir / f"coverage_diagnostic_{diagnostic_date}.md"

    json_report = render_json(classifier_result, simulator_result, metadata, examples_per_bucket=examples_per_bucket)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)
        f.write("\n")

    md_report = render_markdown(classifier_result, simulator_result, metadata, examples_per_bucket=examples_per_bucket)
    md_path.write_text(md_report, encoding="utf-8")

    # Summary print.
    print(f"total_misses: {classifier_result['total_misses']:,}")
    for name in PRIMARY_CAUSE_ORDER:
        b = classifier_result["primary_causes"][name]
        print(f"  {name:>36s}: {b['count']:>6,}  ({b['pct']:.2f}%)")
    print(f"Report: {md_path}")
    print(f"Report: {json_path}")
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description="Sumerian anchor coverage diagnostic")
    parser.add_argument("--anchors", default=str(root / "data" / "processed" / "english_anchors.json"))
    parser.add_argument("--fused",   default=str(root / "models" / "fused_embeddings_1536d.npz"))
    parser.add_argument("--glove",   default=str(root / "data" / "processed" / "glove.6B.300d.txt"))
    parser.add_argument("--gemma",   default=str(root / "models" / "english_gemma_whitened_768d.npz"))
    parser.add_argument("--ridge-gemma", default=str(root / "models" / "ridge_weights_gemma_whitened.npz"))
    parser.add_argument("--oracc-lemmas", default=str(root / "data" / "raw" / "oracc_lemmas.json"))
    parser.add_argument("--cleaned-corpus", default=str(root / "data" / "processed" / "cleaned_corpus.txt"))
    parser.add_argument("--fasttext-model", default=str(root / "models" / "fasttext_sumerian.model"))
    parser.add_argument("--out-dir", default=str(root / "results"))
    parser.add_argument("--date", default=_dt.date.today().isoformat(),
                        help="Diagnostic date (YYYY-MM-DD), used in output filenames")
    parser.add_argument("--examples-per-bucket", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    return run_diagnostic(
        anchors_path=Path(args.anchors),
        fused_path=Path(args.fused),
        glove_path=Path(args.glove),
        gemma_path=Path(args.gemma),
        ridge_gemma_path=Path(args.ridge_gemma),
        oracc_lemmas_path=Path(args.oracc_lemmas),
        cleaned_corpus_path=Path(args.cleaned_corpus),
        fasttext_model_path=Path(args.fasttext_model),
        out_dir=Path(args.out_dir),
        diagnostic_date=args.date,
        examples_per_bucket=args.examples_per_bucket,
    )


if __name__ == "__main__":
    sys.exit(main())
