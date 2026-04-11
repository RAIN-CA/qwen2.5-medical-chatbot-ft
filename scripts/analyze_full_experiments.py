import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("outputs/experiments")
OUT = ROOT / "analysis_full"
OUT.mkdir(parents=True, exist_ok=True)

DOMAINS = ["general", "finance", "legal", "medical"]

# ---------- 自动解析 ----------
def parse_info(p):
    parts = p.parts

    # 根据你当前结构自动推断
    model = parts[-5]
    size = parts[-4]
    setting = parts[-3]

    # dataset 识别（关键新增）
    # 方式1：文件名
    name = p.name
    if "datasetA" in name:
        dataset = "A"
    elif "datasetB" in name:
        dataset = "B"
    else:
        # fallback：目录
        dataset = "default"

    return model, size, setting, dataset

# ---------- 读取 ----------
def load_all():
    rows = []

    for p in ROOT.rglob("all_domains_report.json"):
        if "analysis" in p.parts:
            continue

        model, size, setting, dataset = parse_info(p)

        with open(p) as f:
            d = json.load(f)

        row = {
            "model": model,
            "size": size,
            "dataset": dataset,
            "setting": setting,
            "avg_cosine": d["avg_cosine"],
        }

        for dm in DOMAINS:
            row[f"{dm}_cosine"] = d["by_domain"][dm]["avg_cosine"]

        rows.append(row)

    return pd.DataFrame(rows)

# ---------- Δ计算 ----------
def compute_delta(df):
    out = []

    for (model, size, dataset), sub in df.groupby(["model", "size", "dataset"]):

        base = sub[sub["setting"]=="base"]
        ft   = sub[sub["setting"]!="base"]

        if base.empty or ft.empty:
            continue

        base = base.iloc[0]

        for _, ft_row in ft.iterrows():

            row = {
                "model": model,
                "size": size,
                "dataset": dataset,
                "setting": ft_row["setting"],
                "delta_avg": ft_row["avg_cosine"] - base["avg_cosine"]
            }

            deltas = []
            for d in DOMAINS:
                delta = ft_row[f"{d}_cosine"] - base[f"{d}_cosine"]
                row[f"delta_{d}"] = delta
                deltas.append(delta)

            row["std"] = pd.Series(deltas).std()
            out.append(row)

    return pd.DataFrame(out)

# ---------- 图1：总体提升 ----------
def plot_overall(df):
    plt.figure(figsize=(10,5))

    labels = df["model"] + "_" + df["dataset"]
    plt.bar(labels, df["delta_avg"])

    plt.xticks(rotation=45)
    plt.title("LoRA Improvement Across Models & Datasets")
    plt.ylabel("Δ cosine")

    plt.tight_layout()
    plt.savefig(OUT / "overall_delta.png", dpi=200)
    plt.close()

# ---------- 图2：domain 提升 ----------
def plot_domain(df):
    for dataset in df["dataset"].unique():

        sub = df[df["dataset"]==dataset]
        sub = sub.set_index("model")

        sub[[f"delta_{d}" for d in DOMAINS]].plot(
            kind="bar",
            figsize=(10,6)
        )

        plt.title(f"Domain Improvement - Dataset {dataset}")
        plt.ylabel("Δ cosine")
        plt.tight_layout()
        plt.savefig(OUT / f"domain_delta_{dataset}.png", dpi=200)
        plt.close()

# ---------- 图3：稳定性 ----------
def plot_std(df):
    plt.figure()

    labels = df["model"] + "_" + df["dataset"]
    plt.bar(labels, df["std"])

    plt.xticks(rotation=45)
    plt.title("Stability (std across domains)")
    plt.ylabel("std")

    plt.tight_layout()
    plt.savefig(OUT / "stability.png", dpi=200)
    plt.close()

# ---------- 主函数 ----------
def main():
    df = load_all()
    df.to_csv(OUT / "all_raw.csv", index=False)

    delta = compute_delta(df)
    delta.to_csv(OUT / "delta.csv", index=False)

    plot_overall(delta)
    plot_domain(delta)
    plot_std(delta)

    print("=== DONE ===")
    print(delta.head())

if __name__ == "__main__":
    main()
