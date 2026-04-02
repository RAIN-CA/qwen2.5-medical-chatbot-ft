#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."
export PYTHONPATH=.

TEST_FILE="data/test_sets/all_domains_test.jsonl"

run_eval () {
  local model_family=$1
  local model_size=$2
  local run_type=$3

  local MODEL_DIR="outputs/experiments/${model_family}/${model_size}/${run_type}/adapter"
  local OUT_DIR="outputs/experiments/${model_family}/${model_size}/${run_type}"

  local PRED_FILE="${OUT_DIR}/predictions/all_domains.jsonl"
  local REPORT_FILE="${OUT_DIR}/reports/all_domains_report.json"

  mkdir -p "${OUT_DIR}/predictions"
  mkdir -p "${OUT_DIR}/reports"

  # =========================
  # skip if done
  # =========================
  if [[ -f "$REPORT_FILE" ]]; then
    echo "==============================================="
    echo "SKIP (already done): ${model_family}/${model_size}/${run_type}"
    echo "==============================================="
    return 0
  fi

  if [[ ! -d "$MODEL_DIR" ]]; then
    echo "Model not found, skip: ${MODEL_DIR}"
    return 0
  fi

  echo "==============================================="
  echo "RUN: ${model_family}/${model_size}/${run_type}"
  echo "==============================================="

  # =========================
  # inference
  # =========================
  python -m src.eval_seq2seq.run_inference_on_test_bart \
    --model_path "${MODEL_DIR}" \
    --test_file "${TEST_FILE}" \
    --output_file "${PRED_FILE}" \
    --model_name_for_report "${model_family}_${model_size}_${run_type}"

  # =========================
  # evaluation
  # =========================
  python -m src.eval.evaluate_predictions \
    --pred_file "${PRED_FILE}" \
    --report_file "${REPORT_FILE}"
}

# =========================
# BART
# =========================
run_eval bart base medical_ft
run_eval bart base balanced_multidomain_ft

# =========================
# T5
# =========================
run_eval t5 base medical_ft
run_eval t5 base balanced_multidomain_ft

echo "==============================================="
echo "All seq2seq eval done"
echo "==============================================="
