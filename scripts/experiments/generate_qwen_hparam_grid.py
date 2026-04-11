import itertools
from copy import deepcopy
from pathlib import Path
import yaml

BASE_CONFIG = Path("configs/train/qwen25/0.5b/balanced_multidomain_ft.yaml")
OUT_DIR = Path("configs/train/qwen25/0.5b/hparam_sweeps")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== sweep space =====
LEARNING_RATES = [1e-4, 2e-4, 5e-4]
LORA_RS = [4, 8, 16]
LORA_ALPHAS = [8, 16, 32]
LORA_DROPOUTS = [0.0, 0.05, 0.1]

# Keep training setup fixed in round 1
NUM_EPOCHS = [1]
TRAIN_BATCH_SIZES = [2]
GRAD_ACC_STEPS = [4]

with open(BASE_CONFIG, "r", encoding="utf-8") as f:
    base_cfg = yaml.safe_load(f)

count = 0

for lr, r, alpha, dropout, epochs, bs, ga in itertools.product(
    LEARNING_RATES,
    LORA_RS,
    LORA_ALPHAS,
    LORA_DROPOUTS,
    NUM_EPOCHS,
    TRAIN_BATCH_SIZES,
    GRAD_ACC_STEPS,
):
    cfg = deepcopy(base_cfg)

    exp_name = (
        f"lr{lr:g}"
        f"_r{r}"
        f"_a{alpha}"
        f"_d{str(dropout).replace('.', 'p')}"
        f"_ep{epochs}"
        f"_bs{bs}"
        f"_ga{ga}"
    )

    cfg["training"]["learning_rate"] = float(lr)
    cfg["training"]["num_train_epochs"] = int(epochs)
    cfg["training"]["per_device_train_batch_size"] = int(bs)
    cfg["training"]["gradient_accumulation_steps"] = int(ga)

    cfg["lora"]["r"] = int(r)
    cfg["lora"]["alpha"] = int(alpha)
    cfg["lora"]["dropout"] = float(dropout)

    cfg["training"]["output_dir"] = (
        f"outputs/hparam_sweeps/qwen25/0.5b/balanced_multidomain_ft/{exp_name}"
    )

    cfg["misc"]["report_to"] = "none"

    out_path = OUT_DIR / f"{exp_name}.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    count += 1

print(f"Generated {count} configs in: {OUT_DIR}")
