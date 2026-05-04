"""Shim: re-exports from 10_export_production.py for import-friendly access.

Monkeypatchable: MODELS_DIR, RESULTS_DIR, FINAL_OUTPUT are module-level names.
main() reads them from this module's namespace so tests can patch via monkeypatch.
"""
from importlib.util import spec_from_file_location, module_from_spec
import os
import sys

_spec = spec_from_file_location(
    "export",
    os.path.join(os.path.dirname(__file__), "10_export_production.py"),
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

project_all_vectors = _mod.project_all_vectors
MODELS_DIR = _mod.MODELS_DIR
RESULTS_DIR = _mod.RESULTS_DIR
FINAL_OUTPUT = _mod.FINAL_OUTPUT
SCHEMA_VERSION = _mod.SCHEMA_VERSION


def main():
    """Delegate to _mod.main but inject this module's patchable path attrs first."""
    _this = sys.modules[__name__]
    saved = (_mod.MODELS_DIR, _mod.RESULTS_DIR, _mod.FINAL_OUTPUT)
    _mod.MODELS_DIR = _this.MODELS_DIR
    _mod.RESULTS_DIR = _this.RESULTS_DIR
    _mod.FINAL_OUTPUT = _this.FINAL_OUTPUT
    try:
        _mod.main()
    finally:
        _mod.MODELS_DIR, _mod.RESULTS_DIR, _mod.FINAL_OUTPUT = saved
