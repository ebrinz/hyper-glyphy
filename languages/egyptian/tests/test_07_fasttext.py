import os
import tempfile


def test_corpus_iterator():
    from languages.egyptian.scripts.fasttext_07 import CorpusIterator

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("nTr Hr nb\n")
        f.write("wsjr Ast st\n")
        f.write("ra xpr Htp\n")
        f.flush()

        lines = list(CorpusIterator(f.name))

    os.unlink(f.name)

    assert len(lines) == 3
    assert lines[0] == ["nTr", "Hr", "nb"]
    assert lines[1] == ["wsjr", "Ast", "st"]


def test_train_fasttext_model():
    from languages.egyptian.scripts.fasttext_07 import train_fasttext

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for _ in range(100):
            f.write("nTr Hr nb wsjr Ast st ra xpr Htp mAat\n")
        f.flush()

        with tempfile.TemporaryDirectory() as tmpdir:
            model = train_fasttext(
                corpus_path=f.name,
                output_dir=tmpdir,
                vector_size=32,
                window=5,
                min_count=1,
                epochs=2,
            )

            assert model.vector_size == 32
            assert "nTr" in model.wv

    os.unlink(f.name)
