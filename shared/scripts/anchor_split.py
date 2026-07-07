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


def _ed_le_1(a, b):
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(x != y for x, y in zip(a, b)) == 1
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = j = diff = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            diff += 1
            if diff > 1:
                return False
            j += 1
    return True


def build_groups(anchors, surface_key, fallback="gloss", near_surface_edges=True):
    """Return one group id per anchor via union-find.

    Node keys per anchor: ("surface", <source surface>) always, plus
    ("lemma", l) for each entry in anchor["lemmas"] when present; otherwise
    the fallback node — ("gloss", english) or ("surface_cf", casefolded
    surface). With near_surface_edges, anchors sharing a gloss whose surfaces
    are within edit distance 1 also merge (kills residual spelling-variant
    leakage, e.g. TLHdig cf orthography).
    """
    if fallback not in ("gloss", "surface_casefold"):
        raise ValueError(f"unknown fallback: {fallback}")
    uf = _UnionFind()
    anchor_nodes = []
    for a in anchors:
        surface_node = ("surface", a[surface_key])
        lemmas = a.get("lemmas")
        if lemmas:
            others = [("lemma", l) for l in lemmas]
        elif fallback == "gloss":
            others = [("gloss", a["english"])]
        else:
            others = [("surface_cf", a[surface_key].casefold())]
        for node in others:
            uf.union(surface_node, node)
        anchor_nodes.append(surface_node)

    if near_surface_edges:
        by_gloss = {}
        for a in anchors:
            by_gloss.setdefault(a["english"], set()).add(a[surface_key])
        for gloss, surfaces in by_gloss.items():
            ss = sorted(surfaces)
            for i in range(len(ss)):
                for j in range(i + 1, len(ss)):
                    if _ed_le_1(ss[i], ss[j]):
                        uf.union(("surface", ss[i]), ("surface", ss[j]))

    return [uf.find(n) for n in anchor_nodes]


def group_split(anchors, surface_key, val_size=VAL_SIZE, test_size=TEST_SIZE,
                seed=SEED, fallback="gloss", near_surface_edges=True):
    """Split anchors into (train, val, test); no group spans partitions.

    Deterministic for a given (anchors, seed). Original anchor order is
    preserved within each partition. `fallback` and `near_surface_edges` are
    forwarded to build_groups unchanged.
    """
    if not anchors:
        return [], [], []

    group_ids = build_groups(anchors, surface_key, fallback=fallback,
                             near_surface_edges=near_surface_edges)
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
