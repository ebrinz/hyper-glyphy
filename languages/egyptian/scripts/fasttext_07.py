from importlib.util import spec_from_file_location, module_from_spec
import os

_spec = spec_from_file_location(
    "fasttext",
    os.path.join(os.path.dirname(__file__), "07_train_fasttext.py"),
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

CorpusIterator = _mod.CorpusIterator
train_fasttext = _mod.train_fasttext
