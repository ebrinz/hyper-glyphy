"""Importable shim for 11_fetch_incantations.py (Python can't import names starting with digits)."""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "fetch_incantations",
    os.path.join(os.path.dirname(__file__), "11_fetch_incantations.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

is_incantation = _mod.is_incantation
extract_sux_tokens = _mod.extract_sux_tokens
normalize_docs = _mod.normalize_docs
compute_hit_stats = _mod.compute_hit_stats
parse_incantation_zip = _mod.parse_incantation_zip
load_catalogue = _mod.load_catalogue
MIN_IN_VOCAB_TOKENS = _mod.MIN_IN_VOCAB_TOKENS
