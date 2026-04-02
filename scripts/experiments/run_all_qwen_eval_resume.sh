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

BASE_MODEL_05="Qwen/Qwen2.5-0.5B-Instruct"
BASE_MODEL_3="Qwen/Qwen2.5-3B-Instruct"
TEST_FILE="data/test_sets/all_domains_test.jsonl"

should_run_filter () {
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

run_eval () {
  local family=$1
  local size=$2
  local run_type=$3
  local base_model=$4

  if ! should_run_filter "$size" "$run_type"; then
    return 0
  fi

  local OUT_DIR="outputs/experiments/${family}/${size}/${run_type}"
  local PRED_FILE="${OUT_DIR}/predictions/all_domains.jsonl"
  local REPORT_FILE="${OUT_DIR}/reports/all_domains_report.json"

  mkdir -p "${OUT_DIR}/predictions"
  mkdir -p "${OUT_DIR}/reports"

  if [[ -f "$REPORT_FILE" ]]; then
    echo "=================================================="
    echo "SKIP (already finished): ${family} / ${size} / ${run_type}"
    echo "=================================================="
    return 0
  fi

  echo "=================================================="
  echo "EVAL: ${family} / ${size} / ${run_type}"
  echo "=================================================="

  if [[ "$run_type" == "base" ]]; then
    python -m src.eval.run_inference_on_test \
      --base_model "${base_model}" \
      --test_file "${TEST_FILE}" \
      --output_file "${PRED_FILE}" \
      --model_name_for_report "${family}_${size}_${run_type}"
  else
    local ADAPTER_PATH="${OUT_DIR}/adapter"

    if [[ ! -d "$ADAPTER_PATH" ]]; then
      echo "Adapter path not found: $ADAPTER_PATH"
      exit 1
    fi

    python -m src.eval.run_inference_on_test \
      --base_model "${base_model}" \
      --adapter_path "${ADAPTER_PATH}" \
      --test_file "${TEST_FILE}" \
      --output_file "${PRED_FILE}" \
      --model_name_for_report "${family}_${size}_${run_type}"
  fi

  python -m src.eval.evaluate_predictions \
    --pred_file "${PRED_FILE}" \
    --report_file "${REPORT_FILE}"
}

# ---------- qwen25 0.5b ----------
run_eval qwen25 0.5b base "${BASE_MODEL_05}"
run_eval qwen25 0.5b medical_ft "${BASE_MODEL_05}"
run_eval qwen25 0.5b finance_ft "${BASE_MODEL_05}"
run_eval qwen25 0.5b legal_ft "${BASE_MODEL_05}"
run_eval qwen25 0.5b general_ft "${BASE_MODEL_05}"
run_eval qwen25 0.5b balanced_multidomain_ft "${BASE_MODEL_05}"

# ---------- qwen25 3b ----------
run_eval qwen25 3b base "${BASE_MODEL_3}"
run_eval qwen25 3b medical_ft "${BASE_MODEL_3}"
run_eval qwen25 3b finance_ft "${BASE_MODEL_3}"
run_eval qwen25 3b legal_ft "${BASE_MODEL_3}"
run_eval qwen25 3b general_ft "${BASE_MODEL_3}"
run_eval qwen25 3b balanced_multidomain_ft "${BASE_MODEL_3}"

echo "=================================================="
echo "All requested eval jobs finished."
echo "=================================================="
