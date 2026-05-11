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
    # already had (some 404'd):
    "rinap/rinap1",
    "rinap/rinap3",
    "rinap/rinap4",
    "rinap/rinap5",
    "saao/saa01",
    "saao/saa17",
    "saao/saa18",
    "saao/saa19",
    "cams/akklove",
    "cams/anzu",
    "cams/gkab",
    # NEW: more RINAP/Royal Inscriptions
    "rinap/rinap2",
    "rinap/rinap5p1",
    # NEW: more SAA letters (huge corpus)
    "saao/saa02",
    "saao/saa03",
    "saao/saa04",
    "saao/saa05",
    "saao/saa06",
    "saao/saa07",
    "saao/saa08",
    "saao/saa09",
    "saao/saa10",
    "saao/saa11",
    "saao/saa12",
    "saao/saa13",
    "saao/saa14",
    "saao/saa15",
    "saao/saa16",
    "saao/saa20",
    "saao/saa21",
    # NEW: Royal Inscriptions of Babylonia
    "ribo/babylon2",
    "ribo/babylon3",
    "ribo/babylon4",
    "ribo/babylon5",
    "ribo/babylon6",
    "ribo/babylon7",
    # NEW: Anti-witchcraft rituals
    "cmawro/cmawr1",
    "cmawro/cmawr2",
    "cmawro/maqlu",
    # NEW: Ashurbanipal library
    "asbp/ninmed",
    "asbp/rlasb",
    # NEW: Astronomical Diaries (late Babylonian)
    "adsd/adart1",
    "adsd/adart2",
    "adsd/adart3",
    # NEW: Archive of Texts of Assyrian Empire (many sites)
    "atae/assur",
    "atae/kalhu",
    "atae/nineveh",
    "atae/tushhan",
    # NEW: misc Akkadian projects
    "borsippa",
    "suhu",
    "btmao",
    "btto",
    "iraq/iraq85",
    "glass",
    "amgg",
    "ario",
    "contrib/amarna",
    "contrib/lambert",
    # NEW: more CAMS
    "cams/barutu",
    "cams/ludlul",
    "cams/selbi",
    "cams/tlab",
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
