"""
Document-level evaluation: SIF centroids over aligned word vectors.

Benchmarks: (a) ETCSL genre classification (leave-one-out nearest-centroid),
(b) cross-language parallel retrieval (see parallels subcommand, added later).

See: docs/superpowers/specs/2026-07-06-eval-redesign-design.md

Note (2026-07, A6): document tokenization here is raw line.split() + the
slot normalizer, while FastText corpora pass through each slot's
05_clean_and_tokenize. A known, accepted inconsistency of this parked
doc-level plane — do not "fix" it without re-running Gates 1/2.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from languages.sumerian.scripts.sumerian_normalize import normalize_sumerian_token  # noqa: E402

GENRE_CLASSES = {"1": "narrative", "2": "royal", "4": "hymns",
                 "5": "literature", "6": "proverbs"}
MIN_COMPOSITIONS = 10
SIF_A = 1e-3
_TOKEN_RE = re.compile(r"[a-zA-ZšŠḫḪṭṬṣṢĝĜ]+\d*")

ETCSL_PATH = _ROOT / "languages" / "sumerian" / "data" / "raw" / "etcsl_texts.json"


def _tokenize_sumerian(text):
    return [t for t in (normalize_sumerian_token(m) for m in _TOKEN_RE.findall(text.lower())) if t]


def parse_etcsl_compositions(records):
    """Group ETCSL lines into compositions; genre from catalogue class digit."""
    comps = {}
    for r in records:
        line_id = r.get("line_id", "")
        comp = line_id.split(".")[0]          # 'c2554.A.1' -> 'c2554'
        if len(comp) < 2 or not comp.startswith("c"):
            continue
        genre = GENRE_CLASSES.get(comp[1])
        if genre is None:
            continue
        entry = comps.setdefault(comp, {"genre": genre, "tokens": []})
        entry["tokens"].extend(_tokenize_sumerian(r.get("transliteration", "")))
    return comps


def sif_weights(token_counts, a=SIF_A):
    total = sum(token_counts.values())
    return {w: a / (a + c / total) for w, c in token_counts.items()}


def doc_centroid(tokens, vocab, vectors, weights):
    rows, ws = [], []
    for t in tokens:
        i = vocab.get(t)
        if i is not None:
            rows.append(vectors[i])
            ws.append(weights.get(t, 1.0))
    if not rows:
        return None
    M = np.asarray(rows, dtype=np.float32)
    w = np.asarray(ws, dtype=np.float32)[:, None]
    return (M * w).sum(axis=0) / w.sum()


def loo_nearest_centroid(doc_vecs, labels):
    """Leave-one-out nearest-centroid classification accuracy (cosine)."""
    X = np.asarray(doc_vecs, dtype=np.float32)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    labels = np.asarray(labels)
    sums, counts = {}, {}
    for lab in set(labels):
        sel = labels == lab
        sums[lab] = X[sel].sum(axis=0)
        counts[lab] = int(sel.sum())
    correct = 0
    for i in range(len(X)):
        best_lab, best_sim = None, -2.0
        for lab in sums:
            s, c = sums[lab], counts[lab]
            if labels[i] == lab:
                if c == 1:
                    continue          # singleton class: no LOO centroid
                cent = (s - X[i]) / (c - 1)
            else:
                cent = s / c
            cent = cent / (np.linalg.norm(cent) + 1e-12)
            sim = float(Xn[i] @ cent)
            if sim > best_sim:
                best_lab, best_sim = lab, sim
        correct += best_lab == labels[i]
    return correct / len(X) * 100


def _load_space(npz_path):
    import pickle
    npz_path = Path(npz_path)
    data = np.load(str(npz_path), allow_pickle=True)
    vectors = data["vectors"].astype(np.float32)
    if "vocab" in data:
        vocab_list = data["vocab"]
    else:
        # Derive sidecar name deterministically: strip _gemma_vectors or _vectors suffix,
        # append _vocab, then try .json and .pkl in order.
        stem = npz_path.stem
        base = re.sub(r"_gemma_vectors$|_vectors$", "", stem)
        sidecar_json = npz_path.parent / f"{base}_vocab.json"
        sidecar_pkl = npz_path.parent / f"{base}_vocab.pkl"
        if sidecar_json.exists():
            with open(sidecar_json) as f:
                vocab_list = json.load(f)
        elif sidecar_pkl.exists():
            with open(sidecar_pkl, "rb") as f:
                vocab_list = pickle.load(f)  # project-internal file, not user-supplied
        else:
            raise FileNotFoundError(
                f"No vocab key in {npz_path} and no sidecar found; "
                f"tried: {sidecar_json}, {sidecar_pkl}"
            )
    vocab = {str(w): i for i, w in enumerate(vocab_list)}
    return vocab, vectors


def run_genre():
    with open(ETCSL_PATH) as f:
        comps = parse_etcsl_compositions(json.load(f))
    by_genre = Counter(c["genre"] for c in comps.values())
    comps = {k: v for k, v in comps.items() if by_genre[v["genre"]] >= MIN_COMPOSITIONS}
    counts = Counter(t for c in comps.values() for t in c["tokens"])
    weights = sif_weights(counts)

    sfo = _ROOT / "languages" / "sumerian" / "final_output"
    spaces = {
        "gemma_aligned": sfo / "sumerian_aligned_gemma_vectors.npz",
        "glove_aligned": sfo / "sumerian_aligned_vectors.npz",
        "fused_unaligned": _ROOT / "languages" / "sumerian" / "models" / "fused_embeddings_1536d.npz",
    }
    out = {"n_compositions": len(comps),
           "genre_counts": dict(Counter(c["genre"] for c in comps.values()))}
    for name, path in spaces.items():
        if not path.exists():
            out[name] = "MISSING: " + str(path)
            continue
        vocab, vectors = _load_space(path)
        vecs, labels = [], []
        for c in comps.values():
            v = doc_centroid(c["tokens"], vocab, vectors, weights)
            if v is not None:
                vecs.append(v)
                labels.append(c["genre"])
        out[name] = {"n_docs": len(vecs),
                     "loo_accuracy": round(loo_nearest_centroid(vecs, labels), 2)}
        print(f"{name:<16} n={len(vecs)} LOO acc={out[name]['loo_accuracy']}%")
    majority = max(out["genre_counts"].values()) / out["n_compositions"] * 100
    out["majority_baseline"] = round(majority, 2)
    print(f"majority-class baseline: {majority:.2f}%")
    res = _ROOT / "languages" / "sumerian" / "results" / "doc_eval_genre.json"
    with open(res, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to: {res}")


def mean_reciprocal_rank(ranks):
    return round(sum(1.0 / r for r in ranks) / len(ranks), 4)


def _slot_documents():
    """Per-slot {doc_id: tokens} for the slots with per-text corpora."""
    docs = {}
    with open(ETCSL_PATH) as f:
        comps = parse_etcsl_compositions(json.load(f))
    docs["sumerian"] = {k: v["tokens"] for k, v in comps.items()}

    from languages.hittite.scripts.hittite_normalize import normalize_hittite_token
    from languages.greek.scripts.greek_normalize import normalize_greek_token

    for slot, normalizer in (("hittite", normalize_hittite_token),
                             ("greek", normalize_greek_token)):
        path = _ROOT / "languages" / slot / "data" / "raw" / f"{slot}_texts.json"
        out = {}
        for t in json.load(open(path)):
            toks = [normalizer(w) for line in t["lines"] for w in line.split()]
            out[t["p_number"]] = [tok for tok in toks if tok]
        docs[slot] = out
    return docs


PARALLEL_PAIRS = [
    # (slot_a, doc_matcher_a, slot_b, doc_matcher_b, label)
    # kumarbi-theogony: KUB 33.120 (CTH 344) absent from corpus under that number;
    # rescued via join: KBo 52.10+ (215 lines, contains Alalu→Anu→Kumarbi succession)
    # and KUB 47.56 (50 lines, Kumarbi+Alalu); KBo 52.10+ = join incl. KUB 33.120.
    ("hittite", lambda p: ("KBo 52.10+" in p) or ("KUB 47.56" in p),
     "greek", lambda p: "Theogon" in p, "kumarbi-theogony"),
    ("hittite", lambda p: ("KBo 3.7" in p) or ("KUB 17.5" in p), "greek",
     lambda p: "Theogon" in p, "illuyanka-typhon"),
    # ullikummi-typhon: CTH 345 (Song of Ullikummi) via KBo 26.58 (145 lines) and
    # KBo 26.61 (86 lines); Typhon parallel in Hesiod Theogony.
    ("hittite", lambda p: ("KBo 26.58" in p) or ("KBo 26.61" in p),
     "greek", lambda p: "Theogon" in p, "ullikummi-typhon"),
]


SPACE_NPZ = {"ridge": "{slot}_aligned_gemma_vectors.npz",
             "procrustes": "{slot}_procrustes_gemma_vectors.npz"}
SPACE_RESULTS = {"ridge": "doc_eval_parallels.json",
                 "procrustes": "doc_eval_parallels_procrustes.json"}


def parallel_space_npz(slot, space):
    """final_output npz for a slot under the given alignment space."""
    return (_ROOT / "languages" / slot / "final_output"
            / SPACE_NPZ[space].format(slot=slot))


def run_parallels(space="ridge"):
    docs = _slot_documents()
    aligned = {}
    for slot in docs:
        path = parallel_space_npz(slot, space)
        if path.exists():
            aligned[slot] = _load_space(path)

    # SIF weights per slot from its own document tokens
    weights = {s: sif_weights(Counter(t for d in docs[s].values() for t in d))
               for s in docs}

    centroids = {}
    for slot in aligned:
        vocab, vectors = aligned[slot]
        centroids[slot] = {
            did: v for did, toks in docs[slot].items()
            if (v := doc_centroid(toks, vocab, vectors, weights[slot])) is not None
        }

    results, ranks = [], []
    for slot_a, match_a, slot_b, match_b, label in PARALLEL_PAIRS:
        if slot_a not in centroids or slot_b not in centroids:
            results.append({"pair": label, "status": "DROPPED: missing aligned space"})
            continue
        a_ids = [d for d in centroids[slot_a] if match_a(d)]
        b_ids = [d for d in centroids[slot_b] if match_b(d)]
        if not a_ids or not b_ids:
            results.append({"pair": label, "status":
                            f"DROPPED: unmatched (a={len(a_ids)}, b={len(b_ids)})"})
            continue
        qv = np.mean([centroids[slot_a][d] for d in a_ids], axis=0)
        tv = np.mean([centroids[slot_b][d] for d in b_ids], axis=0)
        # Rank the true target among ALL other-slot documents.
        pool_ids = list(centroids[slot_b].keys())
        pool = np.array([centroids[slot_b][d] for d in pool_ids], dtype=np.float32)
        pool_n = pool / (np.linalg.norm(pool, axis=1, keepdims=True) + 1e-12)
        qn = qv / (np.linalg.norm(qv) + 1e-12)
        sims = pool_n @ qn
        target_sim = float((tv / (np.linalg.norm(tv) + 1e-12)) @ qn)
        rank = int((sims > target_sim).sum()) + 1
        ranks.append(rank)
        results.append({"pair": label, "rank": rank, "pool_size": len(pool_ids),
                        "matched_a": a_ids[:5], "matched_b": b_ids[:5]})
        print(f"{label:<20} rank {rank}/{len(pool_ids)}")

    out = {"space": space, "pairs": results,
           "mrr": mean_reciprocal_rank(ranks) if ranks else None}
    res = _ROOT / "shared" / "results" / SPACE_RESULTS[space]
    res.parent.mkdir(exist_ok=True)
    with open(res, "w") as f:
        json.dump(out, f, indent=2)
    print(f"MRR: {out['mrr']}  Saved to: {res}")


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("benchmark", choices=("genre", "parallels"))
    p.add_argument("--space", choices=("ridge", "procrustes"), default="ridge",
                   help="Alignment space for parallels (genre ignores this)")
    args = p.parse_args()
    if args.benchmark == "genre":
        run_genre()
    else:
        run_parallels(space=args.space)


if __name__ == "__main__":
    main()
