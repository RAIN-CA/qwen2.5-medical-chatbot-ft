import json
import csv
from pathlib import Path

ROOT = Path("outputs/experiments")
rows = []

for model_family in ROOT.iterdir():
    if not model_family.is_dir():
        continue
    for size in model_family.iterdir():
        if not size.is_dir():
            continue
        for run_type in size.iterdir():
            if not run_type.is_dir():
                continue

            report_file = run_type / "reports" / "all_domains_report.json"
            if not report_file.exists():
                continue

            with open(report_file, "r", encoding="utf-8") as f:
                r = json.load(f)

            rows.append({
                "model": model_family.name,
                "size": size.name,
                "run_type": run_type.name,
                "rougeL": r["rouge"].get("rougeL", 0.0),
                "bleu": r.get("avg_bleu", 0.0),
                "cosine": r.get("avg_cosine", 0.0),
            })

out_path = ROOT / "summary.csv"
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "size", "run_type", "rougeL", "bleu", "cosine"])
    writer.writeheader()
    writer.writerows(rows)

print(f"saved: {out_path}")
