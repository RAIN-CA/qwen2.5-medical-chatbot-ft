from datasets import load_dataset
from pathlib import Path

for p in ["data/raw/finance", "data/raw/legal", "data/raw/general"]:
    Path(p).mkdir(parents=True, exist_ok=True)

def run(name, out_dir, config=None, json_url=None):
    print(f"\n===== {name} =====")
    if json_url is not None:
        ds = load_dataset("json", data_files={"train": json_url})
    elif config is None:
        ds = load_dataset(name)
    else:
        ds = load_dataset(name, config)
    ds.save_to_disk(out_dir)
    print(f"Saved to {out_dir}")

# finance
run("vibrantlabsai/fiqa", "data/raw/finance/fiqa_main", config="main")
run("snorkelai/finqa-data", "data/raw/finance/finqa_data")
run("virattt/financial-qa-10K", "data/raw/finance/financial_qa_10k")

# legal
run("isaacus/LegalQAEval", "data/raw/legal/legalqaeval")
run(
    "open-australian-legal-qa-json",
    "data/raw/legal/open_australian_legal_qa",
    json_url="https://huggingface.co/datasets/isaacus/open-australian-legal-qa/resolve/main/qa.jsonl",
)
run("chenghao/sec-material-contracts-qa", "data/raw/legal/sec_material_contracts_qa")

# general
run("rajpurkar/squad", "data/raw/general/squad")
run("rajpurkar/squad_v2", "data/raw/general/squad_v2")
run("hotpotqa/hotpot_qa", "data/raw/general/hotpot_qa", config="fullwiki")

print("\nAll downloads completed.")
