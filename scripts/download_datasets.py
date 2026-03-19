from pathlib import Path
from datasets import load_dataset
import json

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "medquad": {
        "hf_name": "keivalya/MedQuad-MedicalQnADataset",
        "subdir": "medquad",
    },
    "pubmedqa": {
        "hf_name": "bigbio/pubmed_qa",
        "config": "pubmed_qa_labeled_fold0_source",
        "subdir": "pubmedqa",
    },
    "medmcqa": {
        "hf_name": "medmcqa",
        "subdir": "medmcqa",
    },
}

def save_dataset_dict(ds_dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for split, ds in ds_dict.items():
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = split_dir / f"{split}.jsonl"
        parquet_path = split_dir / f"{split}.parquet"
        sample_path = split_dir / f"{split}_sample.json"

        ds.to_json(str(jsonl_path), force_ascii=False)
        ds.to_parquet(str(parquet_path))

        sample_n = min(3, len(ds))
        samples = [ds[i] for i in range(sample_n)]
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

        summary[split] = {
            "num_rows": len(ds),
            "columns": ds.column_names,
            "jsonl": str(jsonl_path),
            "parquet": str(parquet_path),
            "sample": str(sample_path),
        }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

def main():
    for name, meta in DATASETS.items():
        print(f"\n===== Downloading {name} =====")
        hf_name = meta["hf_name"]
        config = meta.get("config", None)
        out_dir = RAW_DIR / meta["subdir"]

        if config is not None:
            ds = load_dataset(hf_name, config, trust_remote_code=True)
        else:
            ds = load_dataset(hf_name, trust_remote_code=True)

        print(ds)
        save_dataset_dict(ds, out_dir)
        print(f"Saved to: {out_dir}")

    print("\nAll datasets downloaded successfully.")

if __name__ == "__main__":
    main()
