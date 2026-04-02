import json
import random
from pathlib import Path

random.seed(42)

ROOT = Path("data/processed")
OUT = Path("data/test_sets")
OUT.mkdir(parents=True, exist_ok=True)


def read_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(items, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def to_eval_format(items, domain):
    out = []
    for item in items:
        msgs = item["messages"]
        user_msg = next(x["content"] for x in msgs if x["role"] == "user")
        assistant_msg = next(x["content"] for x in msgs if x["role"] == "assistant")
        out.append({
            "domain": domain,
            "dataset": item.get("dataset", ""),
            "question": user_msg,
            "reference": assistant_msg,
        })
    return out


medical = read_jsonl(ROOT / "small_test.jsonl")
finance = read_jsonl(ROOT / "finance_small.jsonl")[1200:1400]
legal = read_jsonl(ROOT / "legal_small.jsonl")[1200:1400]
general = read_jsonl(ROOT / "general_small.jsonl")[1200:1400]

medical_eval = to_eval_format(medical[:200], "medical")
finance_eval = to_eval_format(finance[:200], "finance")
legal_eval = to_eval_format(legal[:200], "legal")
general_eval = to_eval_format(general[:200], "general")

write_jsonl(medical_eval, OUT / "medical_test.jsonl")
write_jsonl(finance_eval, OUT / "finance_test.jsonl")
write_jsonl(legal_eval, OUT / "legal_test.jsonl")
write_jsonl(general_eval, OUT / "general_test.jsonl")

all_eval = medical_eval + finance_eval + legal_eval + general_eval
random.shuffle(all_eval)
write_jsonl(all_eval, OUT / "all_domains_test.jsonl")

print({
    "medical": len(medical_eval),
    "finance": len(finance_eval),
    "legal": len(legal_eval),
    "general": len(general_eval),
    "all_domains": len(all_eval),
})
