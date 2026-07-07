"""
Document-level evaluation: SIF centroids over aligned word vectors.

Benchmarks: (a) ETCSL genre classification (leave-one-out nearest-centroid),
(b) cross-language parallel retrieval (see parallels subcommand, added later).

See: docs/superpowers/specs/2026-07-06-eval-redesign-design.md
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
    data = np.load(str(npz_path), allow_pickle=True)
    vectors = data["vectors"].astype(np.float32)
    if "vocab" in data:
        vocab_list = data["vocab"]
    else:
        # Aligned spaces store vocab in a sidecar pkl in the same directory
        pkl_candidates = list(Path(npz_path).parent.glob("*vocab*.pkl"))
        if not pkl_candidates:
            raise FileNotFoundError(f"No vocab key in {npz_path} and no sidecar vocab pkl found")
        with open(pkl_candidates[0], "rb") as f:
            vocab_list = pickle.load(f)  # project-internal file, not user-supplied
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


def run_parallels():
    raise SystemExit("parallels: implemented in Task 10")


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("benchmark", choices=("genre", "parallels"))
    args = p.parse_args()
    if args.benchmark == "genre":
        run_genre()
    else:
        run_parallels()


if __name__ == "__main__":
    main()
