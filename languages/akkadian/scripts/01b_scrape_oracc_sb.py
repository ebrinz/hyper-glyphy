"""
ORACC Standard Babylonian Scraper: Akkadian texts from canonical-descendant
projects (Neo-Assyrian, Neo-Babylonian, Achaemenid). Output is intended for
FastText pretraining ONLY — these projects MUST NOT feed into anchor extraction
because they break the OB temporal-honesty argument.

See: docs/superpowers/specs/2026-05-09-akkadian-slot-design.md
"""
import importlib.util
from pathlib import Path

from tqdm import tqdm

_OB_SCRAPER_PATH = Path(__file__).parent / "01_scrape_oracc_ob.py"
_spec = importlib.util.spec_from_file_location("ob_scraper", _OB_SCRAPER_PATH)
_ob = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ob)

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"

# Standard Babylonian / canonical descendant projects on ORACC.
# Vast amounts of Akkadian here, all linguistically descended from OB.
SB_PROJECTS = [
    "rinap/rinap1",   # Royal Inscriptions of the Neo-Assyrian Period vol 1 (Tiglath-Pileser III)
    "rinap/rinap3",   # Sennacherib
    "rinap/rinap4",   # Esarhaddon
    "rinap/rinap5",   # Ashurbanipal
    "saao/saa01",     # SAA letters vol 1
    "saao/saa17",     # SAA letters vol 17 (Sennacherib correspondence)
    "saao/saa18",
    "saao/saa19",
    "cams/akklove",   # Akkadian Love Literature
    "cams/anzu",      # Anzu epic
    "cams/gkab",      # Geography of Knowledge in Assyria and Babylonia
]


def main():
    out_dir = DATA_RAW / "sb"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_texts, all_lemmas = [], []

    for project in tqdm(SB_PROJECTS, desc="Downloading SB projects"):
        print(f"\nProcessing {project}...")
        zip_path = _ob.download_project_json(project, out_dir)
        if zip_path is None:
            continue
        lemmas, texts = _ob.parse_project_zip(zip_path)
        all_lemmas.extend(lemmas)
        all_texts.extend(texts)
        print(f"  {len(texts)} texts, {len(lemmas)} Akkadian lemmas")

    _ob.save_texts(all_texts, str(DATA_RAW / "sb_texts.json"))
    _ob.save_texts(all_lemmas, str(DATA_RAW / "sb_lemmas.json"))

    total_lines = sum(len(t["lines"]) for t in all_texts)
    print(f"\nTotal SB texts: {len(all_texts)}")
    print(f"Total SB lines: {total_lines}")
    print(f"Total SB Akkadian lemmas: {len(all_lemmas)} (NOT used for anchors)")


if __name__ == "__main__":
    main()
