#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH=.

CONFIG_DIR="configs/train/qwen25/0.5b/hparam_sweeps"
LOG_DIR="outputs/hparam_sweeps/logs"
mkdir -p "$LOG_DIR"

# 用法示例：
#   bash scripts/experiments/run_qwen_hparam_grid.sh
#   bash scripts/experiments/run_qwen_hparam_grid.sh --gpus 0
#   bash scripts/experiments/run_qwen_hparam_grid.sh --gpus 0,1,2,3
#   bash scripts/experiments/run_qwen_hparam_grid.sh --gpus 0,1 --max_jobs 2

GPU_LIST=""
MAX_JOBS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPU_LIST="$2"
      shift 2
      ;;
    --max_jobs)
      MAX_JOBS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

mapfile -t CONFIGS < <(find "$CONFIG_DIR" -maxdepth 1 -name "*.yaml" | sort)

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "No configs found in $CONFIG_DIR"
  exit 1
fi

run_one() {
  local cfg="$1"
  local gpu="$2"

  local name
  name="$(basename "$cfg" .yaml)"
  local log_file="$LOG_DIR/${name}.log"

  # 从 yaml 中读 output_dir
  local out_dir
  out_dir="$(python - <<PY
import yaml
with open("$cfg", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg["training"]["output_dir"])
PY
)"

  # 已完成则跳过
  if [[ -f "${out_dir}/adapter_config.json" || -f "${out_dir}/training_args.bin" ]]; then
    echo "[SKIP] $name already exists at $out_dir"
    return 0
  fi

  echo "=================================================="
  echo "[START] $name"
  echo "Config : $cfg"
  echo "GPU    : ${gpu:-cpu/auto}"
  echo "Output : $out_dir"
  echo "Log    : $log_file"
  echo "=================================================="

  if [[ -n "$gpu" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" python -m src.train.train_lora --config "$cfg" \
      > "$log_file" 2>&1
  else
    python -m src.train.train_lora --config "$cfg" \
      > "$log_file" 2>&1
  fi

  echo "[DONE] $name"
}

# 单任务顺序执行
if [[ -z "$GPU_LIST" || "$MAX_JOBS" -le 1 ]]; then
  for cfg in "${CONFIGS[@]}"; do
    run_one "$cfg" "${GPU_LIST%%,*}"
  done
  exit 0
fi

# 多 GPU 并发执行
IFS=',' read -r -a GPUS <<< "$GPU_LIST"

job_count=0
for idx in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$idx]}"
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"

  run_one "$cfg" "$gpu" &
  ((job_count+=1))

  if (( job_count % MAX_JOBS == 0 )); then
    wait
  fi
done

wait
echo "[ALL DONE]"
