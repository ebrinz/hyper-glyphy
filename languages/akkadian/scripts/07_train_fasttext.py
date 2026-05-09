"""
FastText Training: Train 768d skip-gram embeddings on cleaned Sumerian corpus.

Hyperparameters (from heiroglyphy V15):
  vector_size: 768
  window: 10
  min_count: 5
  epochs: 10
  sg: 1 (skip-gram)
"""
from pathlib import Path

from gensim.models import FastText

DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent.parent / "models"


class CorpusIterator:
    """Iterate over lines in a text file, yielding tokenized lists."""

    def __init__(self, path: str):
        self.path = path

    def __iter__(self):
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:
                    yield tokens


def train_fasttext(
    corpus_path: str,
    output_dir: str,
    vector_size: int = 768,
    window: int = 10,
    min_count: int = 5,
    epochs: int = 10,
) -> FastText:
    """Train FastText skip-gram model on corpus."""
    print(f"Training FastText: dim={vector_size}, window={window}, min_count={min_count}, epochs={epochs}")

    corpus = CorpusIterator(corpus_path)

    model = FastText(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,
        epochs=epochs,
        workers=4,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "fasttext_sumerian.model"
    vec_path = output_dir / "fasttext_sumerian.vec"

    model.save(str(model_path))
    model.wv.save_word2vec_format(str(vec_path))

    print(f"Vocabulary size: {len(model.wv)}")
    print(f"Model saved to: {model_path}")
    print(f"Vectors saved to: {vec_path}")

    return model


def main():
    corpus_path = DATA_PROCESSED / "cleaned_corpus.txt"
    train_fasttext(
        corpus_path=str(corpus_path),
        output_dir=str(MODELS_DIR),
        vector_size=768,
        window=10,
        min_count=5,
        epochs=10,
    )


if __name__ == "__main__":
    main()
