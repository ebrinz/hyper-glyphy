"""
ORACC OB Letters Scraper: Download and parse OB-letter ORACC project JSON archives.

Mirror of 01_scrape_oracc_ob.py. Reuses the CDL walker and downloader.
"""
import importlib.util
from pathlib import Path

from tqdm import tqdm

_OB_SCRAPER_PATH = Path(__file__).parent / "01_scrape_oracc_ob.py"
_spec = importlib.util.spec_from_file_location("ob_scraper", _OB_SCRAPER_PATH)
_ob = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ob)

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"

LETTER_PROJECTS = [
    "saao/saa01",
    "saao/saa17",
    # OB-specific letter projects (Mari ARM, etc.) added as ATF becomes available.
]


def main():
    out_dir = DATA_RAW / "ob_letters"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_texts, all_lemmas = [], []

    for project in tqdm(LETTER_PROJECTS, desc="Downloading OB letter projects"):
        print(f"\nProcessing {project}...")
        zip_path = _ob.download_project_json(project, out_dir)
        if zip_path is None:
            continue
        lemmas, texts = _ob.parse_project_zip(zip_path)
        all_lemmas.extend(lemmas)
        all_texts.extend(texts)
        print(f"  {len(texts)} texts, {len(lemmas)} Akkadian lemmas")

    _ob.save_texts(all_texts, str(DATA_RAW / "ob_letters_texts.json"))
    _ob.save_texts(all_lemmas, str(DATA_RAW / "ob_letters_lemmas.json"))

    total_lines = sum(len(t["lines"]) for t in all_texts)
    print(f"\nTotal letter texts: {len(all_texts)}")
    print(f"Total letter lines: {total_lines}")
    print(f"Total Akkadian letter lemmas: {len(all_lemmas)}")


if __name__ == "__main__":
    main()
