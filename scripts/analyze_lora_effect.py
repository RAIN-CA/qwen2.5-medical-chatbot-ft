import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("outputs/experiments")
OUT_DIR = ROOT / "analysis_lora"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS = ["general", "finance", "legal", "medical"]

# Qwen 内部主分析
QWEN_FAMILY = "qwen25"
QWEN_SETTINGS = [
    "base",
    "balanced_multidomain_ft",
    "general_ft",
    "finance_ft",
    "legal_ft",
    "medical_ft",
]

# 跨架构目前可比的设置
CROSS_ARCH_SETTINGS = [
    "balanced_multidomain_ft",
    "medical_ft",
]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_report_path(report_path: Path):
    """
    支持两种结构：
    1) qwen25/0.5b/base/reports/all_domains_report.json
       => family=qwen25, variant=0.5b, setting=base
    2) bart/base/medical_ft/reports/all_domains_report.json
       => family=bart,   variant=base, setting=medical_ft
    """
    rel = report_path.relative_to(ROOT)
    parts = rel.parts[:-2]  # 去掉 reports/all_domains_report.json

    if len(parts) != 3:
        raise ValueError(f"Unexpected path format: {report_path}")

    family, variant, setting = parts
    return family, variant, setting


def collect_all_reports():
    rows = []
    for report_path in ROOT.rglob("all_domains_report.json"):
        if "analysis" in report_path.parts or "analysis_lora" in report_path.parts:
            continue

        try:
            family, variant, setting = parse_report_path(report_path)
            data = load_json(report_path)

            row = {
                "model_family": family,
                "model_variant": variant,
                "setting": setting,
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
            print(f"[WARN] Skip {report_path}: {e}")

    if not rows:
        raise RuntimeError("No reports found.")

    df = pd.DataFrame(rows)
    return df.sort_values(["model_family", "model_variant", "setting"]).reset_index(drop=True)


def save_all_raw_summary(df):
    out_csv = OUT_DIR / "all_models_raw_summary.csv"
    out_md = OUT_DIR / "all_models_raw_summary.md"

    df.to_csv(out_csv, index=False)

    disp = df.copy()
    for c in disp.columns:
        if disp[c].dtype.kind in "fc":
            disp[c] = disp[c].round(4)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# All Models Raw Summary\n\n")
        f.write(disp.to_markdown(index=False))

    print(f"[OK] {out_csv}")
    print(f"[OK] {out_md}")


# =========================
# Part A: Qwen delta analysis
# =========================

def collect_qwen(df):
    sub = df[
        (df["model_family"] == QWEN_FAMILY) &
        (df["setting"].isin(QWEN_SETTINGS))
    ].copy()

    order = {name: i for i, name in enumerate(QWEN_SETTINGS)}
    sub["setting_order"] = sub["setting"].map(order)
    sub = sub.sort_values(["model_variant", "setting_order"]).reset_index(drop=True)
    return sub


def save_qwen_raw_summary(qwen_df):
    out_csv = OUT_DIR / "qwen_raw_summary.csv"
    out_md = OUT_DIR / "qwen_raw_summary.md"

    qwen_df.to_csv(out_csv, index=False)

    disp = qwen_df.copy()
    for c in disp.columns:
        if disp[c].dtype.kind in "fc":
            disp[c] = disp[c].round(4)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Qwen Raw Summary\n\n")
        f.write(disp.to_markdown(index=False))

    print(f"[OK] {out_csv}")
    print(f"[OK] {out_md}")


def build_qwen_delta_tables(qwen_df):
    abs_rows = []
    delta_rows = []

    for variant in sorted(qwen_df["model_variant"].unique()):
        sub = qwen_df[qwen_df["model_variant"] == variant].copy()
        base = sub[sub["setting"] == "base"]
        if base.empty:
            continue

        base_row = base.iloc[0]

        for _, row in sub.iterrows():
            if row["setting"] == "base":
                continue

            abs_rows.append({
                "model_variant": variant,
                "compare_to": "base",
                "setting": row["setting"],
                "base_avg_cosine": base_row["avg_cosine"],
                "ft_avg_cosine": row["avg_cosine"],
                "base_rougeL": base_row["rougeL"],
                "ft_rougeL": row["rougeL"],
                "base_avg_bleu": base_row["avg_bleu"],
                "ft_avg_bleu": row["avg_bleu"],
            })

            x = {
                "model_variant": variant,
                "setting": row["setting"],
                "delta_avg_cosine": row["avg_cosine"] - base_row["avg_cosine"],
                "delta_rougeL": row["rougeL"] - base_row["rougeL"],
                "delta_avg_bleu": row["avg_bleu"] - base_row["avg_bleu"],
            }
            for d in DOMAINS:
                x[f"delta_{d}_cosine"] = row[f"{d}_cosine"] - base_row[f"{d}_cosine"]
            delta_rows.append(x)

    abs_df = pd.DataFrame(abs_rows)
    delta_df = pd.DataFrame(delta_rows)

    abs_csv = OUT_DIR / "qwen_base_vs_ft_absolute.csv"
    delta_csv = OUT_DIR / "qwen_base_vs_ft_delta.csv"
    abs_md = OUT_DIR / "qwen_base_vs_ft_absolute.md"
    delta_md = OUT_DIR / "qwen_base_vs_ft_delta.md"

    abs_df.to_csv(abs_csv, index=False)
    delta_df.to_csv(delta_csv, index=False)

    with open(abs_md, "w", encoding="utf-8") as f:
        f.write("# Qwen Base vs Fine-tuned Absolute Comparison\n\n")
        x = abs_df.copy()
        for c in x.columns:
            if x[c].dtype.kind in "fc":
                x[c] = x[c].round(4)
        f.write(x.to_markdown(index=False))

    with open(delta_md, "w", encoding="utf-8") as f:
        f.write("# Qwen Fine-tuning Improvement Over Base\n\n")
        x = delta_df.copy()
        for c in x.columns:
            if x[c].dtype.kind in "fc":
                x[c] = x[c].round(4)
        f.write(x.to_markdown(index=False))

    print(f"[OK] {abs_csv}")
    print(f"[OK] {delta_csv}")
    print(f"[OK] {abs_md}")
    print(f"[OK] {delta_md}")

    return abs_df, delta_df


def plot_qwen_overall(qwen_df):
    for variant in sorted(qwen_df["model_variant"].unique()):
        sub = qwen_df[qwen_df["model_variant"] == variant].copy()
        if sub.empty:
            continue

        plt.figure(figsize=(10, 5))
        plt.bar(sub["setting"], sub["avg_cosine"])
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("avg_cosine")
        plt.title(f"Qwen {variant}: Overall Performance Before/After Fine-tuning")
        plt.tight_layout()
        out = OUT_DIR / f"qwen_{variant}_overall_avg_cosine.png"
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"[OK] {out}")


def plot_qwen_domain_delta(delta_df):
    for variant in sorted(delta_df["model_variant"].unique()):
        sub = delta_df[delta_df["model_variant"] == variant].copy()
        if sub.empty:
            continue

        plot_df = sub.set_index("setting")[[f"delta_{d}_cosine" for d in DOMAINS]].copy()
        plot_df.columns = DOMAINS

        ax = plot_df.plot(kind="bar", figsize=(11, 6))
        ax.set_title(f"Qwen {variant}: Domain-wise Cosine Improvement Over Base")
        ax.set_ylabel("Delta Cosine")
        ax.set_xlabel("Fine-tuning Setting")
        plt.axhline(0)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        out = OUT_DIR / f"qwen_{variant}_domain_delta.png"
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"[OK] {out}")


# ===========================================
# Part B: cross-architecture absolute analysis
# ===========================================

def build_cross_arch_tables(df):
    """
    只对共同存在的 setting 做跨架构绝对对比。
    当前重点：
      - balanced_multidomain_ft
      - medical_ft
    """
    cross = df[df["setting"].isin(CROSS_ARCH_SETTINGS)].copy()

    # 生成 label，Qwen 带 variant，其他架构也带 variant
    cross["model_label"] = cross["model_family"] + "_" + cross["model_variant"]

    out_csv = OUT_DIR / "cross_arch_absolute_summary.csv"
    out_md = OUT_DIR / "cross_arch_absolute_summary.md"

    cross.to_csv(out_csv, index=False)

    disp = cross.copy()
    for c in disp.columns:
        if disp[c].dtype.kind in "fc":
            disp[c] = disp[c].round(4)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Cross-Architecture Absolute Comparison\n\n")
        f.write(disp.to_markdown(index=False))

    print(f"[OK] {out_csv}")
    print(f"[OK] {out_md}")
    return cross


def plot_cross_arch_overall(cross_df):
    """
    对每个共同 setting，画 overall avg_cosine/rougeL 柱状图。
    """
    for setting in CROSS_ARCH_SETTINGS:
        sub = cross_df[cross_df["setting"] == setting].copy()
        if sub.empty:
            continue

        # avg_cosine
        plt.figure(figsize=(10, 5))
        plt.bar(sub["model_label"], sub["avg_cosine"])
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("avg_cosine")
        plt.title(f"Cross-Architecture Comparison: {setting} (avg_cosine)")
        plt.tight_layout()
        out1 = OUT_DIR / f"cross_arch_{setting}_avg_cosine.png"
        plt.savefig(out1, dpi=200)
        plt.close()
        print(f"[OK] {out1}")

        # rougeL
        plt.figure(figsize=(10, 5))
        plt.bar(sub["model_label"], sub["rougeL"])
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("rougeL")
        plt.title(f"Cross-Architecture Comparison: {setting} (rougeL)")
        plt.tight_layout()
        out2 = OUT_DIR / f"cross_arch_{setting}_rougeL.png"
        plt.savefig(out2, dpi=200)
        plt.close()
        print(f"[OK] {out2}")


def plot_cross_arch_domain(cross_df):
    """
    对每个共同 setting，画按 domain 的 grouped bar。
    适合看不同架构在某个微调策略下的领域分布差异。
    """
    for setting in CROSS_ARCH_SETTINGS:
        sub = cross_df[cross_df["setting"] == setting].copy()
        if sub.empty:
            continue

        plot_df = sub.set_index("model_label")[[f"{d}_cosine" for d in DOMAINS]].copy()
        plot_df.columns = DOMAINS

        ax = plot_df.plot(kind="bar", figsize=(11, 6))
        ax.set_title(f"Cross-Architecture Domain Comparison: {setting}")
        ax.set_ylabel("Cosine")
        ax.set_xlabel("Model")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        out = OUT_DIR / f"cross_arch_{setting}_domain_cosine.png"
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"[OK] {out}")


def save_cross_arch_best_tables(cross_df):
    """
    每个 setting 下哪个模型整体最好、各领域最好。
    这类更适合表格/markdown。
    """
    lines = ["# Best Model by Cross-Architecture Setting\n"]

    for setting in CROSS_ARCH_SETTINGS:
        sub = cross_df[cross_df["setting"] == setting].copy()
        if sub.empty:
            continue

        lines.append(f"## {setting}\n")

        best_overall = sub.sort_values("avg_cosine", ascending=False).iloc[0]
        lines.append(
            f"- overall avg_cosine best: {best_overall['model_label']} "
            f"(avg_cosine={best_overall['avg_cosine']:.4f}, rougeL={best_overall['rougeL']:.4f})"
        )

        for d in DOMAINS:
            col = f"{d}_cosine"
            best = sub.sort_values(col, ascending=False).iloc[0]
            lines.append(f"- {d} best: {best['model_label']} ({col}={best[col]:.4f})")

        lines.append("")

    out = OUT_DIR / "cross_arch_best_by_setting.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] {out}")


def save_method_note(df, qwen_df):
    families = sorted(df["model_family"].unique())

    text = f"""# Analysis Notes

## Included model families
{", ".join(families)}

## Analysis split

### 1) Qwen delta analysis
Only Qwen has explicit evaluated `setting=base`, so only Qwen supports:
- base vs balanced_multidomain_ft
- base vs general/finance/legal/medical_ft
- delta calculation

### 2) Cross-architecture absolute comparison
For BART/T5, current evaluated settings do not include explicit `setting=base`.
Therefore, for these families we only perform:
- absolute comparison under shared settings
- currently shared settings: {", ".join(CROSS_ARCH_SETTINGS)}

## Recommended presentation

### Use tables for:
- qwen_base_vs_ft_absolute.md
- qwen_base_vs_ft_delta.md
- cross_arch_absolute_summary.md
- cross_arch_best_by_setting.md

### Use figures for:
- qwen_*_overall_avg_cosine.png
- qwen_*_domain_delta.png
- cross_arch_*_avg_cosine.png
- cross_arch_*_rougeL.png
- cross_arch_*_domain_cosine.png

## Interpretation logic
1. First prove whether Qwen improves after LoRA fine-tuning.
2. Then compare whether different Qwen fine-tuning strategies behave differently by domain.
3. Finally compare different transformer architectures under the same fine-tuning setting.
"""
    out = OUT_DIR / "README_analysis.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[OK] {out}")


def main():
    df = collect_all_reports()
    save_all_raw_summary(df)

    # Qwen 主实验
    qwen_df = collect_qwen(df)
    save_qwen_raw_summary(qwen_df)
    _, qwen_delta_df = build_qwen_delta_tables(qwen_df)
    plot_qwen_overall(qwen_df)
    plot_qwen_domain_delta(qwen_delta_df)

    # 跨架构扩展实验
    cross_df = build_cross_arch_tables(df)
    plot_cross_arch_overall(cross_df)
    plot_cross_arch_domain(cross_df)
    save_cross_arch_best_tables(cross_df)

    # 说明
    save_method_note(df, qwen_df)

    print("\n[DONE] Analysis generated at:")
    print(OUT_DIR.resolve())


if __name__ == "__main__":
    main()
