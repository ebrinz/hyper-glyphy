"""
Group-aware train/val/test split for anchor lists.

Prevents surface-variant leakage: all anchors sharing a lemma or a source
surface land in the same partition, so (šarrum, "king") can never train
while (šarru, "king") is tested. Anchors with no `lemmas` field group by
their English gloss instead (Egyptian, whose extraction lives outside this
repo; guarantees no gold label spans the split there).

Assignment is largest-group-first to the partition with the greatest
remaining deficit, so oversized groups (e.g. a high-frequency gloss group)
land in train rather than blowing up the test fraction.

See: docs/superpowers/specs/2026-07-01-lemma-split-eval-design.md
"""
import random

VAL_SIZE = 0.16
TEST_SIZE = 0.20
SEED = 42


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_groups(anchors, surface_key):
    """Return one group id per anchor via union-find.

    Node keys per anchor: ("surface", <source surface>) always, plus
    ("lemma", l) for each entry in anchor["lemmas"] when present, else
    ("gloss", anchor["english"]).
    """
    uf = _UnionFind()
    anchor_nodes = []
    for a in anchors:
        surface_node = ("surface", a[surface_key])
        lemmas = a.get("lemmas")
        if lemmas:
            others = [("lemma", l) for l in lemmas]
        else:
            others = [("gloss", a["english"])]
        for node in others:
            uf.union(surface_node, node)
        anchor_nodes.append(surface_node)
    return [uf.find(n) for n in anchor_nodes]


def group_split(anchors, surface_key, val_size=VAL_SIZE, test_size=TEST_SIZE,
                seed=SEED):
    """Split anchors into (train, val, test); no group spans partitions.

    Deterministic for a given (anchors, seed). Original anchor order is
    preserved within each partition.
    """
    if not anchors:
        return [], [], []

    group_ids = build_groups(anchors, surface_key)
    groups = {}
    for idx, gid in enumerate(group_ids):
        groups.setdefault(gid, []).append(idx)

    members = sorted(groups.values(), key=lambda m: m[0])
    rng = random.Random(seed)
    rng.shuffle(members)
    # Largest first; stable sort keeps the shuffled order among equal sizes.
    members.sort(key=len, reverse=True)

    total = len(anchors)
    targets = {
        "train": (1.0 - val_size - test_size) * total,
        "val": val_size * total,
        "test": test_size * total,
    }
    assigned = {"train": 0, "val": 0, "test": 0}
    out = {"train": [], "val": [], "test": []}
    for m in members:
        part = max(("train", "val", "test"),
                   key=lambda p: targets[p] - assigned[p])
        out[part].extend(m)
        assigned[part] += len(m)

    return (
        [anchors[i] for i in sorted(out["train"])],
        [anchors[i] for i in sorted(out["val"])],
        [anchors[i] for i in sorted(out["test"])],
    )
