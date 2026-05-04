from importlib.util import spec_from_file_location, module_from_spec
import os

_spec = spec_from_file_location(
    "fuse",
    os.path.join(os.path.dirname(__file__), "08_fuse_embeddings.py"),
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

fuse_embeddings = _mod.fuse_embeddings
