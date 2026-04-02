import json
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

print("=" * 110)
print(f"{'model':<12} {'size':<8} {'run_type':<30} {'rougeL':<10} {'bleu':<10} {'cosine':<10}")
print("=" * 110)

for r in sorted(rows, key=lambda x: (x["model"], x["size"], x["run_type"])):
    print(f"{r['model']:<12} {r['size']:<8} {r['run_type']:<30} {r['rougeL']:<10.4f} {r['bleu']:<10.4f} {r['cosine']:<10.4f}")
