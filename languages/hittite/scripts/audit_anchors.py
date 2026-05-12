"""
Akkadian Anchor Quality Audit (Workstream 2a).

Diagnostic-only: classifies every merged anchor produced by
`scripts/06_extract_anchors.py` into mutually-exclusive survival/dropout
buckets against both the GloVe and whitened-Gemma English target vocabs.

Writes a dated report pair under `results/`:
  - anchor_audit_<YYYY-MM-DD>.md
  - anchor_audit_<YYYY-MM-DD>.json  (schema version 1)

See: docs/superpowers/specs/2026-04-18-sumerian-anchor-audit-design.md
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

# Ensure repo root is importable when invoked directly.
_ROOT = _Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import re
from dataclasses import dataclass
from typing import Iterable

# --- Bucket definitions -----------------------------------------------------

BUCKET_ORDER = (
    "junk_hittite",
    "duplicate_collision",
    "low_confidence",
    "hittite_vocab_miss",
    "multiword_english",
    "english_both_miss",
    "english_glove_miss",
    "english_gemma_miss",
    "survives",
)

_MULTIWORD_RE = re.compile(r"[\s_\-]")
AUDIT_SCHEMA_VERSION = 1
DEFAULT_SEED = 42


@dataclass(frozen=True)
class AuditContext:
    fused_vocab: frozenset[str]
    glove_vocab: frozenset[str]
    gemma_vocab: frozenset[str]
    collision_keys: frozenset[str]
    low_conf_threshold: float = 0.3

    def __post_init__(self):
        # Coerce any set-like input to frozenset for true immutability.
        for field_name in ("fused_vocab", "glove_vocab", "gemma_vocab", "collision_keys"):
            value = getattr(self, field_name)
            if not isinstance(value, frozenset):
                object.__setattr__(self, field_name, frozenset(value))


def _normalize_akkadian(raw) -> str:
    if raw is None:
        return ""
    from languages.hittite.scripts.hittite_normalize import normalize_hittite_token
    return normalize_hittite_token(raw)


def _is_junk_hittite(s: str) -> bool:
    if not s:
        return True
    if len(s) <= 1:
        return True
    if not s.strip():
        return True
    return False


def _is_multiword(english_raw: str) -> bool:
    return bool(_MULTIWORD_RE.search(english_raw))


def classify_anchor(anchor: dict, ctx: AuditContext) -> str:
    """Return the bucket name for a single anchor, per priority ordering."""
    hittite_raw = _normalize_akkadian(anchor.get("hittite"))
    english_raw = str(anchor.get("english") or "").strip()
    confidence = float(anchor.get("confidence") or 0.0)

    if _is_junk_hittite(hittite_raw):
        return "junk_hittite"
    if hittite_raw in ctx.collision_keys:
        return "duplicate_collision"
    if confidence < ctx.low_conf_threshold:
        return "low_confidence"
    if hittite_raw not in ctx.fused_vocab:
        return "hittite_vocab_miss"
    if _is_multiword(english_raw):
        return "multiword_english"

    eng_lower = english_raw.lower()
    in_glove = eng_lower in ctx.glove_vocab
    in_gemma = eng_lower in ctx.gemma_vocab
    if not in_glove and not in_gemma:
        return "english_both_miss"
    if not in_glove:
        return "english_glove_miss"
    if not in_gemma:
        return "english_gemma_miss"
    return "survives"


def classify_all(anchors: Iterable[dict], ctx: AuditContext) -> dict:
    """Classify every anchor and return totals + per-bucket rows + Venn counts.

    The `english_venn` counts are computed ONLY over anchors that reached the
    English-side membership check — i.e., the union of the three `english_*_miss`
    buckets plus `survives`. Anchors that early-exited into junk, dedup-collision,
    low-confidence, hittite_vocab_miss, or multiword_english buckets do NOT
    appear in the Venn. Therefore `sum(venn.values())` equals the size of those
    four buckets combined, not `totals.merged`.
    """
    anchors = list(anchors)
    buckets: dict[str, list[dict]] = {name: [] for name in BUCKET_ORDER}

    for anchor in anchors:
        bucket = classify_anchor(anchor, ctx)
        buckets[bucket].append(anchor)

    survives_count = len(buckets["survives"])
    merged = len(anchors)

    venn = {
        "in_glove_in_gemma": 0,
        "in_glove_not_gemma": 0,
        "not_glove_in_gemma": 0,
        "not_glove_not_gemma": 0,
    }
    for bucket_name in ("english_both_miss", "english_glove_miss", "english_gemma_miss", "survives"):
        for anchor in buckets[bucket_name]:
            eng_lower = str(anchor.get("english") or "").lower()
            in_glove = eng_lower in ctx.glove_vocab
            in_gemma = eng_lower in ctx.gemma_vocab
            if in_glove and in_gemma:
                venn["in_glove_in_gemma"] += 1
            elif in_glove and not in_gemma:
                venn["in_glove_not_gemma"] += 1
            elif not in_glove and in_gemma:
                venn["not_glove_in_gemma"] += 1
            else:
                venn["not_glove_not_gemma"] += 1

    return {
        "totals": {
            "merged": merged,
            "survives": survives_count,
            "dropped": merged - survives_count,
        },
        "buckets": {
            name: {
                "count": len(rows),
                "pct_total": (len(rows) / merged * 100.0) if merged else 0.0,
                "rows": rows,
            }
            for name, rows in buckets.items()
        },
        "english_venn": venn,
    }


# --- Rendering --------------------------------------------------------------

import numpy as np  # noqa: E402

RECOVERABILITY = {
    "junk_hittite":       "low — upstream extraction bug",
    "duplicate_collision": "low — dedup by design",
    "low_confidence":      "medium — raise threshold only if false-positive rate checks out",
    "hittite_vocab_miss": "high — candidate driver for Workstream 2b (FastText retrain)",
    "multiword_english":   "medium — cheap phrase-splitter or phrase-embedding could recover some",
    "english_both_miss":   "low — genuinely untranslatable or specialist vocab",
    "english_glove_miss":  "medium — check lemmatization / hyphenation variants",
    "english_gemma_miss":  "medium — same, Gemma-specific",
    "survives":            "n/a — already in the valid-anchor set",
}


def _pick_examples(rows: list[dict], n: int, seed: int) -> list[dict]:
    if not rows:
        return []
    n = min(n, len(rows))
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(rows), size=n, replace=False).tolist())
    return [rows[i] for i in indices]


def render_json(result: dict, metadata: dict, examples_per_bucket: int = 10) -> dict:
    """Serializable report dict matching spec schema version 1."""
    seed = metadata.get("source_artifacts", {}).get("seed", DEFAULT_SEED)
    buckets_out = {}
    for name in BUCKET_ORDER:
        bucket = result["buckets"][name]
        buckets_out[name] = {
            "count": bucket["count"],
            "pct_total": round(bucket["pct_total"], 4),
            "examples": _pick_examples(bucket["rows"], examples_per_bucket, seed),
        }
    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "audit_date": metadata["audit_date"],
        "source_artifacts": metadata["source_artifacts"],
        "totals": result["totals"],
        "buckets": buckets_out,
        "english_venn": result["english_venn"],
    }


def _escape_md_cell(s) -> str:
    """Escape pipe characters so user-supplied fields don't break markdown tables."""
    return str(s).replace("|", r"\|")


def _format_row(row: dict) -> str:
    return (
        f"| {_escape_md_cell(row.get('akkadian', ''))} "
        f"| {_escape_md_cell(row.get('english', ''))} "
        f"| {row.get('confidence', 0):.3f} "
        f"| {_escape_md_cell(row.get('source', ''))} |"
    )


def render_markdown(result: dict, metadata: dict, examples_per_bucket: int = 10) -> str:
    totals = result["totals"]
    merged = totals["merged"]
    survives = totals["survives"]
    dropped = totals["dropped"]
    pct_survives = (survives / merged * 100.0) if merged else 0.0
    pct_dropped = (dropped / merged * 100.0) if merged else 0.0

    lines: list[str] = []
    lines.append(f"# Anchor Audit — {metadata['audit_date']}")
    lines.append("")
    lines.append("Generated by `scripts/audit_anchors.py`. See design spec at "
                 "`docs/superpowers/specs/2026-04-18-akkadian-anchor-audit-design.md`.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Count | % of total |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Total merged anchors | {merged:,} | 100.00% |")
    lines.append(f"| Surviving (both target vocabs) | {survives:,} | {pct_survives:.2f}% |")
    lines.append(f"| Dropped | {dropped:,} | {pct_dropped:.2f}% |")
    lines.append("")

    lines.append("## Dropout by bucket (priority-assigned, mutually exclusive)")
    lines.append("")
    lines.append("| Bucket | Count | % of total | % of dropped | Recoverability |")
    lines.append("|---|---:|---:|---:|---|")
    for name in BUCKET_ORDER:
        bucket = result["buckets"][name]
        count = bucket["count"]
        pct_total = bucket["pct_total"]
        pct_dropped_bucket = (count / dropped * 100.0) if dropped and name != "survives" else 0.0
        lines.append(
            f"| `{name}` | {count:,} | {pct_total:.2f}% | "
            f"{pct_dropped_bucket:.2f}% | {RECOVERABILITY[name]} |"
        )
    lines.append("")

    lines.append("## Cross-cut: English-side Venn")
    lines.append("")
    venn = result["english_venn"]
    lines.append("Over anchors that passed priority 1-5 checks and reached the English-side test.")
    lines.append("")
    lines.append("|  | In GloVe | Not in GloVe |")
    lines.append("|---|---:|---:|")
    lines.append(f"| **In Gemma** | {venn['in_glove_in_gemma']:,} | {venn['not_glove_in_gemma']:,} |")
    lines.append(f"| **Not in Gemma** | {venn['in_glove_not_gemma']:,} | {venn['not_glove_not_gemma']:,} |")
    lines.append("")

    lines.append("## Bucket examples")
    lines.append("")
    lines.append(f"Up to {examples_per_bucket} deterministically-sampled rows per non-empty bucket.")
    lines.append("")
    seed = metadata.get("source_artifacts", {}).get("seed", DEFAULT_SEED)
    for name in BUCKET_ORDER:
        bucket = result["buckets"][name]
        examples = _pick_examples(bucket["rows"], examples_per_bucket, seed)
        if not examples:
            continue
        lines.append(f"### `{name}` ({bucket['count']:,} rows)")
        lines.append("")
        lines.append("| akkadian | english | confidence | source |")
        lines.append("|---|---|---:|---|")
        for row in examples:
            lines.append(_format_row(row))
        lines.append("")

    lines.append("## Recoverability narrative")
    lines.append("")
    lines.append(_recoverability_narrative(result))
    lines.append("")

    return "\n".join(lines)


def _recoverability_narrative(result: dict) -> str:
    buckets = result["buckets"]
    dropped = result["totals"]["dropped"]
    if dropped == 0:
        return "All anchors survived — no dropouts to explain."

    high = buckets["hittite_vocab_miss"]["count"]
    med = (
        buckets["low_confidence"]["count"]
        + buckets["multiword_english"]["count"]
        + buckets["english_glove_miss"]["count"]
        + buckets["english_gemma_miss"]["count"]
    )
    low = (
        buckets["junk_hittite"]["count"]
        + buckets["duplicate_collision"]["count"]
        + buckets["english_both_miss"]["count"]
    )

    def pct(n: int) -> str:
        return f"{n / dropped * 100:.1f}%"

    return (
        f"Of the {dropped:,} dropped anchors, "
        f"roughly {pct(high)} ({high:,}) are high-recoverability — Akkadian-side "
        "vocab gaps likely addressable by Workstream 2b (FastText retrain with "
        "different tokenization). "
        f"Another {pct(med)} ({med:,}) are medium-recoverability — "
        "phrase-splitting, lemmatization/hyphenation checks, or threshold "
        "tuning could recover some. "
        f"The remaining {pct(low)} ({low:,}) are low-recoverability — junk, "
        "by-design dedup losses, or specialist/untranslatable vocabulary."
    )


# --- Artifact loaders -------------------------------------------------------

import argparse  # noqa: E402
import datetime as _dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402


def _assert_lowercase_sample(vocab: list[str], source: str, sample_size: int = 100) -> None:
    sample = vocab[:sample_size]
    for word in sample:
        if word != word.lower():
            raise ValueError(
                f"{source} vocab is not lowercase — saw {word!r}. "
                "Audit expects pre-lowercased caches."
            )


def _load_fused_vocab(path: Path) -> set[str]:
    data = np.load(str(path), allow_pickle=True)
    vectors = data["vectors"]
    vocab = [str(w) for w in data["vocab"]]
    if vectors.shape[1] != 1536:
        raise ValueError(
            f"Fused vocab vectors dim {vectors.shape[1]} != 1536 — "
            "regenerate via scripts/08_fuse_embeddings.py"
        )
    if len(vocab) != vectors.shape[0]:
        raise ValueError("Fused vocab row/vector count mismatch")
    return set(vocab)


def _load_gemma_vocab(path: Path) -> set[str]:
    data = np.load(str(path))
    vectors = data["vectors"]
    vocab = [str(w) for w in data["vocab"]]
    if vectors.shape[1] != 768:
        raise ValueError(
            f"Gemma vocab vectors dim {vectors.shape[1]} != 768 — "
            "regenerate via scripts/whiten_gemma.py"
        )
    if len(vocab) != vectors.shape[0]:
        raise ValueError("Gemma vocab row/vector count mismatch")
    _assert_lowercase_sample(vocab, "Gemma")
    return set(vocab)


def _load_glove_vocab(path: Path) -> set[str]:
    vocab: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.rstrip("\n").split(" ")
            if len(parts) != 301:
                raise ValueError(
                    f"GloVe line {line_no} has {len(parts) - 1} dims, expected 300 — "
                    "verify data/processed/glove.6B.300d.txt integrity"
                )
            vocab.append(parts[0])
    _assert_lowercase_sample(vocab, "GloVe")
    return set(vocab)


def _load_anchors(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        anchors = json.load(f)
    bad_rows: list[tuple[int, dict]] = []
    for i, row in enumerate(anchors):
        if not isinstance(row, dict):
            bad_rows.append((i, row))
            continue
        if "akkadian" not in row or "english" not in row or "confidence" not in row:
            bad_rows.append((i, row))
        if len(bad_rows) >= 5:
            break
    if bad_rows:
        raise ValueError(
            f"Anchors JSON has malformed rows; first bad examples: {bad_rows}"
        )
    return anchors


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _reconstruct_dedup_collisions(
    raw_oracc_path: Path | None,
    raw_etcsl_path: Path | None,
    merged_anchors: list[dict],
) -> set[str]:
    """Return the set of Akkadian keys lost to merge_anchors' higher-confidence
    dedup. If raw inputs are missing, returns an empty set."""
    if raw_oracc_path is None or raw_etcsl_path is None:
        return set()
    if not raw_oracc_path.exists() or not raw_etcsl_path.exists():
        print(
            f"WARN: raw inputs not found at {raw_oracc_path} / {raw_etcsl_path}; "
            "duplicate_collision bucket will be 0.",
            file=sys.stderr,
        )
        return set()

    # The upstream file is named `06_extract_anchors.py`, which isn't directly
    # importable (leading digit). Use importlib.util to load it by path.
    import importlib.util
    root = Path(__file__).parent.parent
    spec_path = root / "scripts" / "06_extract_anchors.py"
    spec = importlib.util.spec_from_file_location("extract_anchors_06_mod", spec_path)
    extract_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extract_module)

    with open(raw_oracc_path) as f:
        lemmas = json.load(f)
    dict_anchors = extract_module.extract_epsd2_anchors(lemmas, min_occurrences=5)

    with open(raw_etcsl_path) as f:
        etcsl_lines = json.load(f)
    parallel = [line for line in etcsl_lines if line.get("translation")]
    cooc_anchors = extract_module.extract_cooccurrence_anchors(
        parallel, min_cooccurrences=3, min_confidence=0.3
    )

    pre_merge_keys = {a["akkadian"] for a in (dict_anchors + cooc_anchors)}
    post_merge_keys = {a["akkadian"] for a in merged_anchors}
    return pre_merge_keys - post_merge_keys


# --- Main -------------------------------------------------------------------


def run_audit(
    *,
    anchors_path: Path,
    fused_path: Path,
    gemma_path: Path,
    glove_path: Path,
    raw_oracc_path: Path | None,
    raw_etcsl_path: Path | None,
    out_dir: Path,
    audit_date: str,
    examples_per_bucket: int = 10,
    seed: int = DEFAULT_SEED,
) -> int:
    anchors = _load_anchors(anchors_path)
    fused_vocab = _load_fused_vocab(fused_path)
    gemma_vocab = _load_gemma_vocab(gemma_path)
    glove_vocab = _load_glove_vocab(glove_path)
    collision_keys = _reconstruct_dedup_collisions(raw_oracc_path, raw_etcsl_path, anchors)

    ctx = AuditContext(
        fused_vocab=fused_vocab,
        glove_vocab=glove_vocab,
        gemma_vocab=gemma_vocab,
        collision_keys=collision_keys,
    )

    result = classify_all(anchors, ctx)

    metadata = {
        "audit_date": audit_date,
        "source_artifacts": {
            "anchors_path": str(anchors_path),
            "anchors_sha256": _sha256(anchors_path),
            "fused_vocab_path": str(fused_path),
            "fused_vocab_size": len(fused_vocab),
            "glove_path": str(glove_path),
            "glove_vocab_size": len(glove_vocab),
            "gemma_path": str(gemma_path),
            "gemma_vocab_size": len(gemma_vocab),
            "seed": seed,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"anchor_audit_{audit_date}.json"
    md_path = out_dir / f"anchor_audit_{audit_date}.md"

    json_report = render_json(result, metadata, examples_per_bucket=examples_per_bucket)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)
        f.write("\n")

    md_report = render_markdown(result, metadata, examples_per_bucket=examples_per_bucket)
    md_path.write_text(md_report, encoding="utf-8")

    totals = result["totals"]
    pct = (totals["survives"] / totals["merged"] * 100.0) if totals["merged"] else 0.0
    print(
        f"survives: {totals['survives']:,}/{totals['merged']:,} ({pct:.2f}%), "
        f"dropped: {totals['dropped']:,}"
    )
    print(f"Report: {md_path}")
    print(f"Report: {json_path}")
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description="Akkadian anchor quality audit")
    parser.add_argument(
        "--anchors",
        default=str(root / "data" / "processed" / "english_anchors.json"),
    )
    parser.add_argument(
        "--fused",
        default=str(root / "models" / "fused_embeddings_1536d.npz"),
    )
    parser.add_argument(
        "--gemma",
        default=str(root / "models" / "english_gemma_whitened_768d.npz"),
    )
    parser.add_argument(
        "--glove",
        default=str(root / "data" / "processed" / "glove.6B.300d.txt"),
    )
    parser.add_argument(
        "--raw-oracc",
        default=str(root / "data" / "raw" / "ob_literary_lemmas.json"),
    )
    parser.add_argument(
        "--raw-etcsl",
        default=None,
    )
    parser.add_argument(
        "--out-dir",
        default=str(root / "results"),
    )
    parser.add_argument(
        "--date",
        default=_dt.date.today().isoformat(),
        help="Audit date (YYYY-MM-DD), used in output filenames",
    )
    parser.add_argument("--examples-per-bucket", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    raw_oracc_path = Path(args.raw_oracc) if args.raw_oracc else None
    raw_etcsl_path = Path(args.raw_etcsl) if args.raw_etcsl else None
    return run_audit(
        anchors_path=Path(args.anchors),
        fused_path=Path(args.fused),
        gemma_path=Path(args.gemma),
        glove_path=Path(args.glove),
        raw_oracc_path=raw_oracc_path,
        raw_etcsl_path=raw_etcsl_path,
        out_dir=Path(args.out_dir),
        audit_date=args.date,
        examples_per_bucket=args.examples_per_bucket,
    )


if __name__ == "__main__":
    sys.exit(main())
