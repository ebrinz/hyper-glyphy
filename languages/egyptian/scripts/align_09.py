from importlib.util import spec_from_file_location, module_from_spec
import os

_spec = spec_from_file_location(
    "align",
    os.path.join(os.path.dirname(__file__), "09_align_and_evaluate.py"),
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

build_training_data = _mod.build_training_data
train_ridge = _mod.train_ridge
evaluate_alignment = _mod.evaluate_alignment
select_alpha = _mod.select_alpha
filter_stopword_glosses = _mod.filter_stopword_glosses
val_top1_csls = _mod.val_top1_csls if hasattr(_mod, "val_top1_csls") else None
