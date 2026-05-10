import json
import tempfile

import numpy as np
import pytest


# --- Classification tests ---------------------------------------------------


def _default_ctx(
    fused_vocab=None,
    glove_vocab=None,
    gemma_vocab=None,
    collision_keys=None,
    low_conf_threshold=0.3,
):
    from languages.akkadian.scripts.audit_anchors import AuditContext

    return AuditContext(
        fused_vocab=set(fused_vocab or {"lugal", "dingir"}),
        glove_vocab=set(glove_vocab or {"king", "god"}),
        gemma_vocab=set(gemma_vocab or {"king", "god"}),
        collision_keys=set(collision_keys or set()),
        low_conf_threshold=low_conf_threshold,
    )


def test_survives_requires_both_target_vocabs():
    from languages.akkadian.scripts.audit_anchors import classify_anchor

    ctx = _default_ctx(glove_vocab={"king"}, gemma_vocab={"king", "god"})
    bucket = classify_anchor(
        {"akkadian": "lugal", "english": "king", "confidence": 0.9, "source": "ePSD2"}, ctx
    )
    assert bucket == "survives"

    bucket2 = classify_anchor(
        {"akkadian": "dingir", "english": "god", "confidence": 0.9, "source": "ePSD2"}, ctx
    )
    assert bucket2 == "english_glove_miss"


def test_bucket_priority_ordering_akkadian_miss_beats_multiword():
    from languages.akkadian.scripts.audit_anchors import classify_anchor

    ctx = _default_ctx(fused_vocab={"dingir"})
    bucket = classify_anchor(
        {"akkadian": "e2.gal", "english": "royal palace", "confidence": 0.9, "source": "ePSD2"},
        ctx,
    )
    assert bucket == "akkadian_vocab_miss"


def test_junk_akkadian_detection():
    from languages.akkadian.scripts.audit_anchors import classify_anchor

    ctx = _default_ctx()
    for bad in ["", " ", "\t", "a", "\u200b"]:
        bucket = classify_anchor(
            {"akkadian": bad, "english": "king", "confidence": 0.9, "source": "ePSD2"}, ctx
        )
        assert bucket == "junk_akkadian", f"expected junk for {bad!r}, got {bucket}"


def test_low_confidence_bucket():
    from languages.akkadian.scripts.audit_anchors import classify_anchor

    ctx = _default_ctx(low_conf_threshold=0.3)
    bucket = classify_anchor(
        {"akkadian": "lugal", "english": "king", "confidence": 0.25, "source": "ETCSL"}, ctx
    )
    assert bucket == "low_confidence"

    # Confidence exactly at threshold is NOT low_confidence (strict <).
    at_threshold = classify_anchor(
        {"akkadian": "lugal", "english": "king", "confidence": 0.3, "source": "ETCSL"}, ctx
    )
    assert at_threshold != "low_confidence"


def test_duplicate_collision_bucket():
    from languages.akkadian.scripts.audit_anchors import classify_anchor

    ctx = _default_ctx(collision_keys={"lugal"})
    bucket = classify_anchor(
        {"akkadian": "lugal", "english": "monarch", "confidence": 0.5, "source": "ETCSL"}, ctx
    )
    assert bucket == "duplicate_collision"


def test_multiword_english_detection():
    from languages.akkadian.scripts.audit_anchors import classify_anchor

    ctx = _default_ctx()
    for phrase in ["royal palace", "to cut off", "sun-god", "cut_off"]:
        bucket = classify_anchor(
            {"akkadian": "lugal", "english": phrase, "confidence": 0.9, "source": "ePSD2"},
            ctx,
        )
        assert bucket == "multiword_english", f"expected multiword for {phrase!r}, got {bucket}"


def test_english_both_miss_when_neither_vocab_has_word():
    from languages.akkadian.scripts.audit_anchors import classify_anchor

    ctx = _default_ctx(glove_vocab={"god"}, gemma_vocab={"god"})
    bucket = classify_anchor(
        {"akkadian": "lugal", "english": "king", "confidence": 0.9, "source": "ePSD2"}, ctx
    )
    assert bucket == "english_both_miss"


def test_english_gemma_miss_when_only_glove_has_word():
    from languages.akkadian.scripts.audit_anchors import classify_anchor

    ctx = _default_ctx(glove_vocab={"king"}, gemma_vocab={"god"})
    bucket = classify_anchor(
        {"akkadian": "lugal", "english": "king", "confidence": 0.9, "source": "ePSD2"}, ctx
    )
    assert bucket == "english_gemma_miss"


def test_english_glove_miss_when_only_gemma_has_word():
    from languages.akkadian.scripts.audit_anchors import classify_anchor

    ctx = _default_ctx(glove_vocab={"god"}, gemma_vocab={"king"})
    bucket = classify_anchor(
        {"akkadian": "lugal", "english": "king", "confidence": 0.9, "source": "ePSD2"}, ctx
    )
    assert bucket == "english_glove_miss"


def test_english_lookup_is_case_insensitive():
    from languages.akkadian.scripts.audit_anchors import classify_anchor

    ctx = _default_ctx(glove_vocab={"king"}, gemma_vocab={"king"})
    bucket = classify_anchor(
        {"akkadian": "lugal", "english": "KING", "confidence": 0.9, "source": "ePSD2"}, ctx
    )
    assert bucket == "survives"


def test_classify_all_sums_to_total():
    from languages.akkadian.scripts.audit_anchors import classify_all

    ctx = _default_ctx(fused_vocab={"lugal"}, glove_vocab={"king"}, gemma_vocab={"king"})
    anchors = [
        {"akkadian": "lugal", "english": "king", "confidence": 0.9, "source": "ePSD2"},
        {"akkadian": "missing", "english": "god", "confidence": 0.9, "source": "ePSD2"},
        {"akkadian": "", "english": "king", "confidence": 0.9, "source": "ePSD2"},
    ]
    result = classify_all(anchors, ctx)
    total = sum(b["count"] for b in result["buckets"].values())
    assert total == len(anchors) == 3
    assert result["buckets"]["survives"]["count"] == 1
    assert result["buckets"]["akkadian_vocab_miss"]["count"] == 1
    assert result["buckets"]["junk_akkadian"]["count"] == 1


def test_classify_all_empty_input():
    from languages.akkadian.scripts.audit_anchors import classify_all

    ctx = _default_ctx()
    result = classify_all([], ctx)
    assert result["totals"]["merged"] == 0
    assert result["totals"]["survives"] == 0
    assert result["totals"]["dropped"] == 0
    for bucket in result["buckets"].values():
        assert bucket["count"] == 0


def test_venn_accounting_over_single_token_anchors():
    from languages.akkadian.scripts.audit_anchors import classify_all

    ctx = _default_ctx(
        fused_vocab={"s1", "s2", "s3", "s4"},
        glove_vocab={"a", "b"},
        gemma_vocab={"a", "c"},
    )
    anchors = [
        {"akkadian": "s1", "english": "a", "confidence": 0.9, "source": "ePSD2"},
        {"akkadian": "s2", "english": "b", "confidence": 0.9, "source": "ePSD2"},
        {"akkadian": "s3", "english": "c", "confidence": 0.9, "source": "ePSD2"},
        {"akkadian": "s4", "english": "d", "confidence": 0.9, "source": "ePSD2"},
    ]
    result = classify_all(anchors, ctx)
    venn = result["english_venn"]
    assert venn["in_glove_in_gemma"] == 1
    assert venn["in_glove_not_gemma"] == 1
    assert venn["not_glove_in_gemma"] == 1
    assert venn["not_glove_not_gemma"] == 1


# --- Render tests ----------------------------------------------------------


def _default_metadata():
    return {
        "audit_date": "2026-04-18",
        "source_artifacts": {
            "anchors_path": "data/processed/english_anchors.json",
            "anchors_sha256": "0" * 64,
            "fused_vocab_path": "models/fused_embeddings_1536d.npz",
            "fused_vocab_size": 2,
            "glove_path": "data/processed/glove.6B.300d.txt",
            "glove_vocab_size": 2,
            "gemma_path": "models/english_gemma_whitened_768d.npz",
            "gemma_vocab_size": 2,
            "seed": 42,
        },
    }


def test_render_json_schema_version_and_structure():
    from languages.akkadian.scripts.audit_anchors import classify_all, render_json

    ctx = _default_ctx()
    anchors = [{"akkadian": "lugal", "english": "king", "confidence": 0.9, "source": "ePSD2"}]
    result = classify_all(anchors, ctx)
    report = render_json(result, _default_metadata(), examples_per_bucket=1)

    assert report["audit_schema_version"] == 1
    assert report["audit_date"] == "2026-04-18"
    assert "source_artifacts" in report
    assert report["totals"]["merged"] == 1
    assert set(report["buckets"].keys()) == {
        "junk_akkadian", "duplicate_collision", "low_confidence",
        "akkadian_vocab_miss", "multiword_english", "english_both_miss",
        "english_glove_miss", "english_gemma_miss", "survives",
    }
    for bucket in report["buckets"].values():
        assert "count" in bucket
        assert "pct_total" in bucket
        assert "examples" in bucket
        assert isinstance(bucket["examples"], list)


def test_render_json_is_deterministic_with_seed():
    from languages.akkadian.scripts.audit_anchors import classify_all, render_json

    ctx = _default_ctx(fused_vocab={f"s{i}" for i in range(20)})
    anchors = [
        {"akkadian": f"s{i}", "english": f"e{i}", "confidence": 0.9, "source": "ePSD2"}
        for i in range(20)
    ]
    result = classify_all(anchors, ctx)
    report_a = render_json(result, _default_metadata(), examples_per_bucket=5)
    report_b = render_json(result, _default_metadata(), examples_per_bucket=5)

    import json as _json
    assert _json.dumps(report_a, sort_keys=True) == _json.dumps(report_b, sort_keys=True)


def test_render_json_examples_respect_per_bucket_cap():
    from languages.akkadian.scripts.audit_anchors import classify_all, render_json

    ctx = _default_ctx(fused_vocab={f"s{i}" for i in range(20)})
    anchors = [
        {"akkadian": f"s{i}", "english": "king", "confidence": 0.9, "source": "ePSD2"}
        for i in range(20)
    ]
    result = classify_all(anchors, ctx)
    report = render_json(result, _default_metadata(), examples_per_bucket=3)
    assert len(report["buckets"]["survives"]["examples"]) == 3


def test_render_markdown_contains_required_sections():
    from languages.akkadian.scripts.audit_anchors import classify_all, render_markdown

    ctx = _default_ctx()
    anchors = [{"akkadian": "lugal", "english": "king", "confidence": 0.9, "source": "ePSD2"}]
    result = classify_all(anchors, ctx)
    md = render_markdown(result, _default_metadata(), examples_per_bucket=1)

    assert "# Anchor Audit" in md
    assert "## Summary" in md
    assert "## Dropout by bucket" in md
    assert "## Cross-cut: English-side Venn" in md
    assert "## Bucket examples" in md
    assert "## Recoverability narrative" in md
    for bucket_name in ("junk_akkadian", "survives", "akkadian_vocab_miss"):
        assert bucket_name in md


def test_render_markdown_recoverability_heuristics_present():
    from languages.akkadian.scripts.audit_anchors import classify_all, render_markdown

    ctx = _default_ctx()
    anchors = [{"akkadian": "lugal", "english": "king", "confidence": 0.9, "source": "ePSD2"}]
    result = classify_all(anchors, ctx)
    md = render_markdown(result, _default_metadata(), examples_per_bucket=1)

    assert "high" in md.lower() or "medium" in md.lower() or "low" in md.lower()


def test_render_markdown_escapes_pipe_characters_in_cells():
    from languages.akkadian.scripts.audit_anchors import classify_all, render_markdown

    ctx = _default_ctx(fused_vocab={"|an.usan|"}, glove_vocab={"night"}, gemma_vocab={"night"})
    anchors = [
        {"akkadian": "|an.usan|", "english": "night", "confidence": 0.9, "source": "ePSD2|ETCSL"},
    ]
    result = classify_all(anchors, ctx)
    md = render_markdown(result, _default_metadata(), examples_per_bucket=1)

    # Raw pipes from the fields should not appear unescaped in the examples table.
    # The escaped form `\|` should be present instead.
    assert r"\|an.usan\|" in md
    assert r"ePSD2\|ETCSL" in md


# --- Loader and main integration tests -------------------------------------


def _write_anchors(tmp_path, rows):
    path = tmp_path / "english_anchors.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    return path


def _write_fused_vocab(tmp_path, vocab):
    path = tmp_path / "fused_embeddings_1536d.npz"
    np.savez_compressed(
        str(path),
        vectors=np.zeros((len(vocab), 1536), dtype=np.float32),
        vocab=np.array(vocab),
    )
    return path


def _write_gemma_vocab(tmp_path, vocab):
    path = tmp_path / "english_gemma_whitened_768d.npz"
    np.savez_compressed(
        str(path),
        vectors=np.zeros((len(vocab), 768), dtype=np.float32),
        vocab=np.array(vocab),
    )
    return path


def _write_glove(tmp_path, vocab):
    path = tmp_path / "glove.6B.300d.txt"
    with open(path, "w", encoding="utf-8") as f:
        for word in vocab:
            zeros = " ".join(["0.0"] * 300)
            f.write(f"{word} {zeros}\n")
    return path


def test_load_fused_vocab_enforces_shape():
    from languages.akkadian.scripts.audit_anchors import _load_fused_vocab

    with tempfile.TemporaryDirectory() as tmp:
        import pathlib
        tmp_path = pathlib.Path(tmp)
        bad = tmp_path / "bad.npz"
        np.savez_compressed(
            str(bad),
            vectors=np.zeros((3, 999), dtype=np.float32),
            vocab=np.array(["a", "b", "c"]),
        )
        with pytest.raises(ValueError, match="1536"):
            _load_fused_vocab(bad)


def test_load_glove_vocab_reads_first_token_per_line():
    from languages.akkadian.scripts.audit_anchors import _load_glove_vocab

    with tempfile.TemporaryDirectory() as tmp:
        import pathlib
        tmp_path = pathlib.Path(tmp)
        path = _write_glove(tmp_path, ["king", "god", "palace"])
        vocab = _load_glove_vocab(path)
        assert vocab == {"king", "god", "palace"}


def test_load_glove_vocab_detects_malformed_row():
    from languages.akkadian.scripts.audit_anchors import _load_glove_vocab

    with tempfile.TemporaryDirectory() as tmp:
        import pathlib
        tmp_path = pathlib.Path(tmp)
        path = tmp_path / "bad.txt"
        with open(path, "w") as f:
            f.write("king " + " ".join(["0.0"] * 299) + "\n")
        with pytest.raises(ValueError, match="300"):
            _load_glove_vocab(path)


def test_main_writes_report_pair(tmp_path):
    import languages.akkadian.scripts.audit_anchors as mod

    anchors = [
        {"akkadian": "lugal", "english": "king", "confidence": 0.9, "source": "ePSD2"},
        {"akkadian": "", "english": "junk", "confidence": 0.9, "source": "ePSD2"},
    ]
    anchors_path = _write_anchors(tmp_path, anchors)
    fused_path = _write_fused_vocab(tmp_path, ["lugal"])
    gemma_path = _write_gemma_vocab(tmp_path, ["king"])
    glove_path = _write_glove(tmp_path, ["king"])

    out_dir = tmp_path / "results"
    out_dir.mkdir()

    exit_code = mod.run_audit(
        anchors_path=anchors_path,
        fused_path=fused_path,
        gemma_path=gemma_path,
        glove_path=glove_path,
        raw_oracc_path=None,
        raw_etcsl_path=None,
        out_dir=out_dir,
        audit_date="2026-04-18",
    )

    assert exit_code == 0
    md_path = out_dir / "anchor_audit_2026-04-18.md"
    json_path = out_dir / "anchor_audit_2026-04-18.json"
    assert md_path.exists()
    assert json_path.exists()

    report = json.loads(json_path.read_text())
    assert report["audit_schema_version"] == 1
    assert report["totals"]["merged"] == 2
    assert report["totals"]["survives"] == 1
    assert report["buckets"]["junk_akkadian"]["count"] == 1


def test_main_raises_on_missing_input(tmp_path):
    import languages.akkadian.scripts.audit_anchors as mod

    with pytest.raises(FileNotFoundError):
        mod.run_audit(
            anchors_path=tmp_path / "missing.json",
            fused_path=tmp_path / "missing.npz",
            gemma_path=tmp_path / "missing.npz",
            glove_path=tmp_path / "missing.txt",
            raw_oracc_path=None,
            raw_etcsl_path=None,
            out_dir=tmp_path,
            audit_date="2026-04-18",
        )


def test_vocab_lowercase_sample_check_raises_on_mixed_case(tmp_path):
    import languages.akkadian.scripts.audit_anchors as mod

    path = tmp_path / "bad_case.npz"
    np.savez_compressed(
        str(path),
        vectors=np.zeros((5, 768), dtype=np.float32),
        vocab=np.array(["KING", "god", "Palace", "water", "sky"]),
    )
    with pytest.raises(ValueError, match="lowercase"):
        mod._load_gemma_vocab(path)
