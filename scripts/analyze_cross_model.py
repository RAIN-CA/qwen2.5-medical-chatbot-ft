import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("outputs/experiments")
OUT_DIR = ROOT / "analysis_cross_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS = ["general", "finance", "legal", "medical"]

def parse_info(p):
    parts = p.parts
    # 通用解析
    model = parts[-5]
    size = parts[-4]
    setting = parts[-3]
    return model, size, setting

def load_all():
    rows = []
    for p in ROOT.rglob("all_domains_report.json"):
        if "analysis" in p.parts:
            continue

        model, size, setting = parse_info(p)

        # 只保留有 base 和 ft 的
        if setting not in [
            "base",
            "balanced_multidomain_ft"
        ]:
            continue

        with open(p) as f:
            d = json.load(f)

        row = {
            "model": model,
            "size": size,
            "setting": setting,
            "avg_cosine": d["avg_cosine"],
        }

        for dm in DOMAINS:
            row[f"{dm}_cosine"] = d["by_domain"][dm]["avg_cosine"]

        rows.append(row)

    return pd.DataFrame(rows)

def compute_delta(df):
    out = []

    for (model, size), sub in df.groupby(["model", "size"]):
        base = sub[sub["setting"]=="base"]
        ft   = sub[sub["setting"]=="balanced_multidomain_ft"]

        if base.empty or ft.empty:
            continue

        base = base.iloc[0]
        ft   = ft.iloc[0]

        row = {
            "model": model,
            "size": size,
            "delta_avg_cosine": ft["avg_cosine"] - base["avg_cosine"],
        }

        deltas = []
        for d in DOMAINS:
            delta = ft[f"{d}_cosine"] - base[f"{d}_cosine"]
            row[f"delta_{d}"] = delta
            deltas.append(delta)

        row["std_across_domain"] = pd.Series(deltas).std()
        out.append(row)

    return pd.DataFrame(out)

def plot_overall_delta(df):
    plt.figure()
    plt.bar(df["model"] + "_" + df["size"], df["delta_avg_cosine"])
    plt.xticks(rotation=45)
    plt.title("LoRA Improvement (avg_cosine)")
    plt.ylabel("Δ cosine")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "delta_overall.png", dpi=200)
    plt.close()

def plot_domain_delta(df):
    df = df.set_index("model")
    df[[f"delta_{d}" for d in DOMAINS]].plot(kind="bar", figsize=(10,6))
    plt.title("Domain-wise LoRA Improvement")
    plt.ylabel("Δ cosine")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "delta_by_domain.png", dpi=200)
    plt.close()

def main():
    df = load_all()
    delta = compute_delta(df)

    delta.to_csv(OUT_DIR / "delta_summary.csv", index=False)

    plot_overall_delta(delta)
    plot_domain_delta(delta)

    print(delta)

if __name__ == "__main__":
    main()
