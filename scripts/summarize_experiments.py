import os
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path("outputs/experiments")
OUT_DIR = ROOT / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS = ["general", "finance", "legal", "medical"]


def parse_experiment_info(report_path: Path):
    """
    从路径中解析实验信息。
    典型路径:
    outputs/experiments/qwen25/0.5b/base/reports/all_domains_report.json
    outputs/experiments/qwen25/0.5b/medical_ft/reports/all_domains_report.json
    outputs/experiments/qwen25/0.5b/balanced_multidomain_ft/reports/all_domains_report.json
    outputs/experiments/bart/base/medical_ft/reports/all_domains_report.json
    """
    rel = report_path.relative_to(ROOT)
    parts = rel.parts

    # 去掉末尾 reports/all_domains_report.json
    # 剩余类似:
    # qwen25 / 0.5b / base
    # bart / base / medical_ft
    exp_parts = parts[:-2]

    model_family = exp_parts[0] if len(exp_parts) >= 1 else "unknown"

    # 针对 qwen25 这类带 size 的
    if model_family == "qwen25":
        model_size = exp_parts[1] if len(exp_parts) >= 2 else "unknown"
        setting = exp_parts[2] if len(exp_parts) >= 3 else "unknown"
    else:
        # bart/t5 当前结构更像 model_family/base/setting
        # 这里把第二层记作 size_or_variant，第三层记作 setting
        model_size = exp_parts[1] if len(exp_parts) >= 2 else "unknown"
        setting = exp_parts[2] if len(exp_parts) >= 3 else "unknown"

    return model_family, model_size, setting


def load_one_report(report_path: Path):
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_family, model_size, setting = parse_experiment_info(report_path)

    row = {
        "model_family": model_family,
        "model_size": model_size,
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

    return row


def collect_reports():
    rows = []
    for report_path in ROOT.rglob("all_domains_report.json"):
        # 跳过 analysis 目录下的意外文件
        if "analysis" in report_path.parts:
            continue
        try:
            rows.append(load_one_report(report_path))
        except Exception as e:
            print(f"[WARN] Failed to parse {report_path}: {e}")

    if not rows:
        raise RuntimeError("No all_domains_report.json files found under outputs/experiments")

    df = pd.DataFrame(rows)

    # 排序便于阅读
    df = df.sort_values(
        by=["model_family", "model_size", "setting"],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    return df


def save_csv_and_markdown(df: pd.DataFrame):
    csv_path = OUT_DIR / "summary_all.csv"
    md_path = OUT_DIR / "summary_table.md"

    df.to_csv(csv_path, index=False)

    # 更适合直接展示的核心表
    show_cols = [
        "model_family", "model_size", "setting",
        "num_samples", "rouge1", "rouge2", "rougeL", "avg_bleu", "avg_cosine",
        "general_cosine", "finance_cosine", "legal_cosine", "medical_cosine"
    ]
    display_df = df[show_cols].copy()

    # 四舍五入更适合汇报
    for col in display_df.columns:
        if display_df[col].dtype.kind in "fc":
            display_df[col] = display_df[col].round(4)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Experiment Summary Table\n\n")
        f.write(display_df.to_markdown(index=False))

    print(f"[OK] Saved CSV: {csv_path}")
    print(f"[OK] Saved Markdown table: {md_path}")


def plot_overall_metric_bar(df: pd.DataFrame):
    """
    用柱状图对比 overall 指标：
    - avg_cosine
    - rougeL
    - avg_bleu

    适合做总体性能横向对比。
    """
    exp_name = df["model_family"] + "_" + df["model_size"] + "_" + df["setting"]

    metrics = [
        ("avg_cosine", "Overall Average Cosine"),
        ("rougeL", "Overall ROUGE-L"),
        ("avg_bleu", "Overall Average BLEU"),
    ]

    for metric, title in metrics:
        plt.figure(figsize=(12, 6))
        plt.bar(exp_name, df[metric])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(metric)
        plt.title(title)
        plt.tight_layout()
        out = OUT_DIR / f"{metric}_bar.png"
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"[OK] Saved plot: {out}")


def plot_domain_cosine_grouped(df: pd.DataFrame):
    """
    用分组柱状图展示各实验在不同 domain 上的 cosine。
    domain 维度最适合图，因为一眼就能看泛化差异。
    """
    plot_df = df.copy()
    plot_df["exp_name"] = (
        plot_df["model_family"] + "_" + plot_df["model_size"] + "_" + plot_df["setting"]
    )

    melted = plot_df.melt(
        id_vars=["exp_name"],
        value_vars=[f"{d}_cosine" for d in DOMAINS],
        var_name="domain",
        value_name="cosine",
    )
    melted["domain"] = melted["domain"].str.replace("_cosine", "", regex=False)

    pivot = melted.pivot(index="exp_name", columns="domain", values="cosine")

    ax = pivot.plot(kind="bar", figsize=(14, 7))
    ax.set_title("Domain-wise Cosine Comparison")
    ax.set_ylabel("Cosine")
    ax.set_xlabel("Experiment")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out = OUT_DIR / "domain_cosine_grouped.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[OK] Saved plot: {out}")


def plot_heatmap_like_table(df: pd.DataFrame):
    """
    不用 seaborn，直接用 matplotlib 的 table，
    生成一个适合截图/汇报的热区表替代图。
    这种形式适合 domain x experiment 的紧凑对比。
    """
    table_df = df.copy()
    table_df["experiment"] = (
        table_df["model_family"] + "\n" + table_df["model_size"] + "\n" + table_df["setting"]
    )
    table_df = table_df[
        ["experiment"] + [f"{d}_cosine" for d in DOMAINS] + ["avg_cosine", "rougeL", "avg_bleu"]
    ].copy()

    for col in table_df.columns[1:]:
        table_df[col] = table_df[col].round(4)

    fig, ax = plt.subplots(figsize=(14, max(4, 0.6 * len(table_df) + 1)))
    ax.axis("off")

    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)

    plt.title("Compact Experiment Comparison Table", pad=20)
    plt.tight_layout()
    out = OUT_DIR / "comparison_table.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved plot-table: {out}")


def save_best_models_table(df: pd.DataFrame):
    """
    表格比画图更适合 '最佳模型' 排名。
    """
    rank_df = df[
        ["model_family", "model_size", "setting", "avg_cosine", "rougeL", "avg_bleu"]
    ].copy()

    rank_df = rank_df.sort_values(by=["avg_cosine", "rougeL", "avg_bleu"], ascending=False)
    rank_df.to_csv(OUT_DIR / "ranking_by_avg_cosine.csv", index=False)

    with open(OUT_DIR / "ranking_by_avg_cosine.md", "w", encoding="utf-8") as f:
        f.write("# Ranking by avg_cosine\n\n")
        f.write(rank_df.round(4).to_markdown(index=False))

    print(f"[OK] Saved ranking tables")


def save_domain_best_tables(df: pd.DataFrame):
    """
    各领域最佳结果更适合表格，不适合单独画很多图。
    """
    lines = ["# Best Experiment by Domain\n"]

    for d in DOMAINS:
        col = f"{d}_cosine"
        best = df.sort_values(by=col, ascending=False).iloc[0]
        lines.append(f"## {d}\n")
        lines.append(
            f"- model_family: {best['model_family']}\n"
            f"- model_size: {best['model_size']}\n"
            f"- setting: {best['setting']}\n"
            f"- {col}: {best[col]:.4f}\n"
            f"- avg_cosine: {best['avg_cosine']:.4f}\n"
            f"- rougeL: {best['rougeL']:.4f}\n"
            f"- avg_bleu: {best['avg_bleu']:.4f}\n"
        )

    with open(OUT_DIR / "best_by_domain.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] Saved best_by_domain.md")


def main():
    df = collect_reports()

    # 原始全量汇总
    save_csv_and_markdown(df)

    # 排名类：更适合表格
    save_best_models_table(df)
    save_domain_best_tables(df)

    # 总体对比：更适合柱状图
    plot_overall_metric_bar(df)

    # 领域泛化：更适合 grouped bar chart
    plot_domain_cosine_grouped(df)

    # 紧凑总览：适合做图表型总表
    plot_heatmap_like_table(df)

    print("\n[DONE] All summaries, tables, and plots saved to:")
    print(OUT_DIR.resolve())


if __name__ == "__main__":
    main()
