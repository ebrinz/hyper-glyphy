from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
_spec = spec_from_file_location("findings_figures",
                                str(_ROOT / "shared" / "scripts" / "findings_figures.py"))
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_four_figures_produced(tmp_path):
    paths = _mod.main(tmp_path)
    assert len(paths) == 4
    names = {p.name for p in paths}
    assert names == {"fig1_procrustes.pdf", "fig2_akkadian_alpha.pdf",
                     "fig3_rsa_matrix.pdf", "fig4_vrtra_control.pdf"}
    for p in paths:
        assert p.exists() and p.stat().st_size > 1000
