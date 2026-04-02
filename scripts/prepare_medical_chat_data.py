from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.builders import build_medical_datasets, build_general_datasets

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def main():
    medical_summary = build_medical_datasets(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        random_seed=42,
    )

    general_summary = build_general_datasets(
        processed_dir=PROCESSED_DIR,
        random_seed=42,
    )

    print("Saved datasets.")
    print(json.dumps({
        "medical": medical_summary,
        "general": general_summary,
    }, indent=2))


if __name__ == "__main__":
    main()
