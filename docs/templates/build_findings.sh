#!/bin/bash
# Build a findings PDF from markdown via the house pandoc/XeLaTeX template.
# Usage (from repo root): bash docs/templates/build_findings.sh <in.md> <out.pdf>
set -euo pipefail
IN=${1:?usage: build_findings.sh <in.md> <out.pdf>}
OUT=${2:?usage: build_findings.sh <in.md> <out.pdf>}
pandoc "$IN" -o "$OUT" \
  --template=docs/templates/hyper-glyphy-pandoc.tex \
  --pdf-engine=xelatex \
  -V mainfont="Times New Roman" \
  -V monofont="Menlo" \
  -V fontsize=11pt \
  -V geometry:margin=1.1in \
  --toc --number-sections
echo "built: $OUT"
