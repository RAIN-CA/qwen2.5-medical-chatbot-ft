#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."
export PYTHONPATH=.

TEST_FILE="data/test_sets/all_domains_test.jsonl"

# medical_ft
python -m src.eval_seq2seq.run_inference_on_test_bart \
  --model_path outputs/experiments/bart/base/medical_ft/adapter \
  --test_file "${TEST_FILE}" \
  --output_file outputs/experiments/bart/base/medical_ft/predictions/all_domains.jsonl \
  --model_name_for_report bart_base_medical_ft

python -m src.eval.evaluate_predictions \
  --pred_file outputs/experiments/bart/base/medical_ft/predictions/all_domains.jsonl \
  --report_file outputs/experiments/bart/base/medical_ft/reports/all_domains_report.json

# balanced_multidomain_ft
python -m src.eval_seq2seq.run_inference_on_test_bart \
  --model_path outputs/experiments/bart/base/balanced_multidomain_ft/adapter \
  --test_file "${TEST_FILE}" \
  --output_file outputs/experiments/bart/base/balanced_multidomain_ft/predictions/all_domains.jsonl \
  --model_name_for_report bart_base_balanced_multidomain_ft

python -m src.eval.evaluate_predictions \
  --pred_file outputs/experiments/bart/base/balanced_multidomain_ft/predictions/all_domains.jsonl \
  --report_file outputs/experiments/bart/base/balanced_multidomain_ft/reports/all_domains_report.json
