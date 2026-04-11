import json
from pathlib import Path
import pandas as pd

ROOT = Path("outputs/experiments")
OUT_DIR = ROOT / "analysis_index"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS = ["general", "finance", "legal", "medical"]

PRIMARY_FAMILIES = {"qwen25"}
COMPARE_ONLY_FAMILIES = {"bart", "t5", "xlnet"}


def parse_report_path(report_path: Path):
    """
    支持路径格式：
    1) outputs/experiments/qwen25/0.5b/base/reports/all_domains_report.json
       -> family=qwen25, variant=0.5b, setting=base

    2) outputs/experiments/bart/base/medical_ft/reports/all_domains_report.json
       -> family=bart, variant=base, setting=medical_ft
    """
    rel = report_path.relative_to(ROOT)
    parts = rel.parts[:-2]  # 去掉 reports/all_domains_report.json

    if len(parts) != 3:
        raise ValueError(f"Unexpected report path format: {report_path}")

    family, variant, setting = parts
    return family, variant, setting


def classify_family(family: str):
    if family in PRIMARY_FAMILIES:
        return True, "primary_qwen"
    if family in COMPARE_ONLY_FAMILIES:
        return False, "baseline_compare_only"
    return False, "other"


def load_report(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_rows():
    rows = []

    for report_path in ROOT.rglob("all_domains_report.json"):
        if "analysis" in report_path.parts or "analysis_lora" in report_path.parts or "analysis_index" in report_path.parts:
            continue

        try:
            family, variant, setting = parse_report_path(report_path)
            is_primary_model, analysis_group = classify_family(family)
            data = load_report(report_path)

            row = {
                "model_family": family,
                "model_variant": variant,
                "setting": setting,
                "is_primary_model": is_primary_model,
                "analysis_group": analysis_group,
                "num_samples": data.get("num_samples"),
                "rouge1": data.get("rouge", {}).get("rouge1"),
                "rouge2": data.get("rouge", {}).get("rouge2"),
                "rougeL": data.get("rouge", {}).get("rougeL"),
                "rougeLsum": data.get("rouge", {}).get("rougeLsum"),
                "avg_bleu": data.get("avg_bleu"),
                "avg_cosine": data.get("avg_cosine"),
                "report_path": str(report_path),
            }

            by_domain = data.get("by_domain", {})
            for d in DOMAINS:
                row[f"{d}_count"] = by_domain.get(d, {}).get("count")
                row[f"{d}_bleu"] = by_domain.get(d, {}).get("avg_bleu")
                row[f"{d}_cosine"] = by_domain.get(d, {}).get("avg_cosine")

            rows.append(row)

        except Exception as e:
            print(f"[WARN] skip {report_path}: {e}")

    if not rows:
        raise RuntimeError("No all_domains_report.json found.")

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["analysis_group", "model_family", "model_variant", "setting"],
        ascending=[True, True, True, True]
    ).reset_index(drop=True)

    return df


def save_outputs(df: pd.DataFrame):
    full_csv = OUT_DIR / "experiment_index_full.csv"
    full_md = OUT_DIR / "experiment_index_full.md"

    qwen_csv = OUT_DIR / "experiment_index_qwen_only.csv"
    qwen_md = OUT_DIR / "experiment_index_qwen_only.md"

    compare_csv = OUT_DIR / "experiment_index_compare_only.csv"
    compare_md = OUT_DIR / "experiment_index_compare_only.md"

    df.to_csv(full_csv, index=False)

    qwen_df = df[df["analysis_group"] == "primary_qwen"].copy()
    compare_df = df[df["analysis_group"] == "baseline_compare_only"].copy()

    qwen_df.to_csv(qwen_csv, index=False)
    compare_df.to_csv(compare_csv, index=False)

    def write_md(path: Path, title: str, frame: pd.DataFrame):
        show = frame.copy()
        for c in show.columns:
            if c in show.columns and hasattr(show[c], "dtype") and show[c].dtype.kind in "fc":
                show[c] = show[c].round(4)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            f.write(show.to_markdown(index=False))

    write_md(full_md, "Experiment Index Full", df)
    write_md(qwen_md, "Experiment Index Qwen Only", qwen_df)
    write_md(compare_md, "Experiment Index Compare Only", compare_df)

    print(f"[OK] {full_csv}")
    print(f"[OK] {full_md}")
    print(f"[OK] {qwen_csv}")
    print(f"[OK] {qwen_md}")
    print(f"[OK] {compare_csv}")
    print(f"[OK] {compare_md}")


def save_readme(df: pd.DataFrame):
    family_counts = df.groupby(["analysis_group", "model_family"]).size().reset_index(name="num_experiments")
    lines = [
        "# Analysis Index Notes",
        "",
        "## Grouping rule",
        "- qwen25 -> primary_qwen",
        "- bart/t5/xlnet -> baseline_compare_only",
        "",
        "## Purpose",
        "- experiment_index_full.csv: all indexed experiments",
        "- experiment_index_qwen_only.csv: main analysis set",
        "- experiment_index_compare_only.csv: comparison-only models",
        "",
        "## Experiment counts by family",
        "",
    ]
    for _, r in family_counts.iterrows():
        lines.append(f"- {r['analysis_group']} / {r['model_family']}: {r['num_experiments']}")

    out = OUT_DIR / "README.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] {out}")


def main():
    df = collect_rows()
    save_outputs(df)
    save_readme(df)
    print("\n[DONE] Indexed experiment files saved to:")
    print(OUT_DIR.resolve())


if __name__ == "__main__":
    main()
