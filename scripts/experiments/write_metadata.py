import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_family", required=True)
    parser.add_argument("--model_size", required=True)
    parser.add_argument("--run_type", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--val_file", required=True)
    args = parser.parse_args()

    out_dir = Path("outputs/experiments") / args.model_family / args.model_size / args.run_type
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "model_family": args.model_family,
        "model_size": args.model_size,
        "run_type": args.run_type,
        "base_model": args.base_model,
        "train_file": args.train_file,
        "val_file": args.val_file,
    }

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved metadata to {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
