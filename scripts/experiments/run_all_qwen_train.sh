#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."
export PYTHONPATH=.

SIZE_FILTER=""
RUN_TYPE_FILTER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --size)
      SIZE_FILTER="$2"
      shift 2
      ;;
    --run_type)
      RUN_TYPE_FILTER="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

should_run () {
  local size="$1"
  local run_type="$2"

  if [[ -n "$SIZE_FILTER" && "$SIZE_FILTER" != "$size" ]]; then
    return 1
  fi

  if [[ -n "$RUN_TYPE_FILTER" && "$RUN_TYPE_FILTER" != "$run_type" ]]; then
    return 1
  fi

  return 0
}

run_train () {
  local family=$1
  local size=$2
  local run_type=$3

  if ! should_run "$size" "$run_type"; then
    return 0
  fi

  echo "=================================================="
  echo "TRAINING: ${family} / ${size} / ${run_type}"
  echo "=================================================="

  python -m src.experiment.run_experiment \
    --stage train \
    --model_family "${family}" \
    --model_size "${size}" \
    --run_type "${run_type}"
}

# qwen25 0.5b
run_train qwen25 0.5b medical_ft
run_train qwen25 0.5b finance_ft
run_train qwen25 0.5b legal_ft
run_train qwen25 0.5b general_ft
run_train qwen25 0.5b balanced_multidomain_ft

# qwen25 3b
run_train qwen25 3b medical_ft
run_train qwen25 3b finance_ft
run_train qwen25 3b legal_ft
run_train qwen25 3b general_ft
run_train qwen25 3b balanced_multidomain_ft
