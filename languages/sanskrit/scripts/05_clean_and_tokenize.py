"""
Corpus Tokenization: normalize the DCS FORM stream for FastText.

Deliberately NOT a clone of the Greek/Sumerian 05 — that script is an ATF
transliteration cleaner (hyphen morpheme-splitting, all-caps sign-name drops,
leading-apostrophe token drops) which would corrupt IAST text (e.g. delete
avagraha forms like 'bravīt). DCS conllu FORM tokens are already clean,
sandhi-resolved IAST words; the only work is canonical normalization.

Output: cleaned_corpus.txt (one line per chapter text, space-separated tokens)
"""
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from languages.sanskrit.scripts.sanskrit_normalize import normalize_sanskrit_token  # noqa: E402

DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"


def clean_line(line: str) -> str:
    """Normalize each whitespace-separated token; drop empties."""
    tokens = [normalize_sanskrit_token(tok) for tok in line.split()]
    return " ".join(t for t in tokens if t)


def build_corpus(texts: list[dict]) -> list[str]:
    """One corpus line per text (all its lines joined), matching the
    format 07's CorpusIterator expects."""
    corpus_lines = []
    for text in texts:
        cleaned_words = []
        for line in text.get("lines", []):
            cleaned = clean_line(line)
            if cleaned:
                cleaned_words.append(cleaned)
        if cleaned_words:
            corpus_lines.append(" ".join(cleaned_words))
    return corpus_lines


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    with open(DATA_PROCESSED / "merged_corpus.json") as f:
        texts = json.load(f)

    corpus_lines = build_corpus(texts)

    output_path = DATA_PROCESSED / "cleaned_corpus.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for line in corpus_lines:
            f.write(line + "\n")

    total_tokens = sum(len(line.split()) for line in corpus_lines)
    vocab = set()
    for line in corpus_lines:
        vocab.update(line.split())

    print(f"Corpus lines: {len(corpus_lines)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Unique tokens: {len(vocab)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
