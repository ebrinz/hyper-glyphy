from shared.scripts.anchor_split import group_split


def _partition_sets(train, val, test, field):
    def collect(part):
        out = set()
        for a in part:
            v = a.get(field)
            if isinstance(v, list):
                out.update(v)
            elif v is not None:
                out.add(v)
        return out
    return collect(train), collect(val), collect(test)


def _make_anchors(n_groups=300, seed_sizes=(1, 1, 1, 2, 2, 3, 5, 8)):
    """Synthetic anchors: n_groups lemmas, varying surface counts per lemma."""
    anchors = []
    for g in range(n_groups):
        size = seed_sizes[g % len(seed_sizes)]
        for s in range(size):
            anchors.append({
                "akkadian": f"lemma{g}_surf{s}",
                "english": f"gloss{g % 50}",
                "lemmas": [f"lemma{g}"],
            })
    return anchors


def test_no_lemma_spans_partitions():
    anchors = _make_anchors()
    train, val, test = group_split(anchors, surface_key="akkadian")
    tr, va, te = _partition_sets(train, val, test, "lemmas")
    assert not (tr & va) and not (tr & te) and not (va & te)


def test_no_surface_spans_partitions():
    # Same surface registered under two different lemmas must not split.
    anchors = _make_anchors()
    anchors.append({"akkadian": "lemma0_surf0", "english": "other",
                    "lemmas": ["lemmaX"]})
    train, val, test = group_split(anchors, surface_key="akkadian")
    tr, va, te = _partition_sets(train, val, test, "akkadian")
    assert not (tr & va) and not (tr & te) and not (va & te)


def test_shared_surface_merges_lemma_groups():
    anchors = [
        {"akkadian": "aaa", "english": "one", "lemmas": ["L1"]},
        {"akkadian": "aaa", "english": "two", "lemmas": ["L2"]},
        {"akkadian": "bbb", "english": "three", "lemmas": ["L2"]},
    ]
    train, val, test = group_split(anchors, surface_key="akkadian")
    parts = [p for p in (train, val, test) if p]
    assert len(parts) == 1 and len(parts[0]) == 3


def test_gloss_fallback_no_gloss_spans_partitions():
    # Anchors without a `lemmas` field group by their English gloss.
    anchors = [{"egyptian_raw": f"w{i}", "english": f"g{i % 40}"}
               for i in range(400)]
    train, val, test = group_split(anchors, surface_key="egyptian_raw")
    tr, va, te = _partition_sets(train, val, test, "english")
    assert not (tr & va) and not (tr & te) and not (va & te)


def test_deterministic():
    anchors = _make_anchors()
    a = group_split(anchors, surface_key="akkadian")
    b = group_split(anchors, surface_key="akkadian")
    assert a == b


def test_partition_proportions():
    anchors = _make_anchors(n_groups=500)
    train, val, test = group_split(anchors, surface_key="akkadian")
    total = len(anchors)
    assert len(train) + len(val) + len(test) == total
    assert abs(len(test) / total - 0.20) < 0.05
    assert abs(len(val) / total - 0.16) < 0.05


def test_empty_input():
    assert group_split([], surface_key="akkadian") == ([], [], [])
