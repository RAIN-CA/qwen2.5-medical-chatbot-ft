import json
import random
from pathlib import Path

random.seed(42)
root = Path("data/processed")

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

medical = read_jsonl(root / "small_train.jsonl")
finance = read_jsonl(root / "finance_small.jsonl")
legal = read_jsonl(root / "legal_small.jsonl")
general = read_jsonl(root / "general_small.jsonl")

target = min(len(medical), len(finance), len(legal), len(general))
print("balanced train target per domain =", target)

random.shuffle(medical)
random.shuffle(finance)
random.shuffle(legal)
random.shuffle(general)

balanced_train = (
    medical[:target] +
    finance[:target] +
    legal[:target] +
    general[:target]
)
random.shuffle(balanced_train)
write_jsonl(balanced_train, root / "balanced_multidomain_train.jsonl")

medical_val = read_jsonl(root / "small_validation.jsonl")
finance_val = read_jsonl(root / "finance_small.jsonl")[1500:1700]
legal_val = read_jsonl(root / "legal_small.jsonl")[1500:1700]
general_val = read_jsonl(root / "general_small.jsonl")[1500:1700]

target_val = min(len(medical_val), len(finance_val), len(legal_val), len(general_val))
print("balanced val target per domain =", target_val)

random.shuffle(medical_val)
random.shuffle(finance_val)
random.shuffle(legal_val)
random.shuffle(general_val)

balanced_val = (
    medical_val[:target_val] +
    finance_val[:target_val] +
    legal_val[:target_val] +
    general_val[:target_val]
)
random.shuffle(balanced_val)
write_jsonl(balanced_val, root / "balanced_multidomain_validation.jsonl")

print({
    "train_total": len(balanced_train),
    "val_total": len(balanced_val),
    "train_per_domain": target,
    "val_per_domain": target_val,
})
