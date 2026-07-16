#!/bin/bash
# Bolt-on: populate the gitignored artifact paths from the companion HF repo
# (https://huggingface.co/datasets/ebrinz/hyper-glyphy-artifacts), then
# recreate the local symlinks the pipeline scripts expect.
#
# Usage (from repo root):  bash shared/scripts/fetch_artifacts.sh
# Requires the `hf` CLI (https://hf.co/cli). Idempotent; safe to re-run.
# Raw third-party corpora are NOT mirrored — see languages/*/data/raw/README.md.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)

hf download ebrinz/hyper-glyphy-artifacts --repo-type dataset --local-dir .

# Per-slot symlinks to the shared English caches (stored once in shared/models).
CACHES="english_gemma_768d.npz english_gemma_bare_768d.npz \
english_gemma_whitened_768d.npz english_gemma_bare_whitened_768d.npz \
gemma_whitening_transform.npz gemma_bare_whitening_transform.npz"
for slot in sumerian akkadian hittite greek sanskrit; do
  mkdir -p "$ROOT/languages/$slot/models"
  for f in $CACHES; do
    ln -sf "$ROOT/shared/models/$f" "$ROOT/languages/$slot/models/$f"
  done
done

# GloVe cache lives once under sumerian; other slots reference it by symlink.
GLOVE="$ROOT/languages/sumerian/data/processed/glove.6B.300d.txt"
for slot in akkadian hittite greek egyptian sanskrit; do
  mkdir -p "$ROOT/languages/$slot/data/processed"
  ln -sf "$GLOVE" "$ROOT/languages/$slot/data/processed/glove.6B.300d.txt"
done

echo "Artifacts fetched and symlinks recreated. Raw corpora (if needed):"
echo "see languages/<slot>/data/raw/README.md for per-source fetch steps."
