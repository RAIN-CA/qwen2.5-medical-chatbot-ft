#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."
export PYTHONPATH=.

echo "===== T5 medical_ft ====="
python -m src.train_seq2seq.train_bart \
  --config configs/train/t5/base/medical_ft.yaml

echo "===== T5 balanced_multidomain_ft ====="
python -m src.train_seq2seq.train_bart \
  --config configs/train/t5/base/balanced_multidomain_ft.yaml
