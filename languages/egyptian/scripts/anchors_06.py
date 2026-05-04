from importlib.util import spec_from_file_location, module_from_spec
import os

_spec = spec_from_file_location(
    "anchors",
    os.path.join(os.path.dirname(__file__), "06_extract_anchors.py"),
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

normalize_anchors = _mod.normalize_anchors
