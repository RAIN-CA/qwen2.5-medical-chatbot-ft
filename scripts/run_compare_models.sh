#!/usr/bin/env bash
set -e

conda activate qwen25-medchat

python -m src.eval.compare_models \
  --model_config configs/eval/model_compare.json \
  --question_file data/eval/medical_questions.txt \
  --output_path outputs/model_comparison_results.json \
  --domain medical
