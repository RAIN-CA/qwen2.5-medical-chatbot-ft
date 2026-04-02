#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."
export PYTHONPATH=.

python -m src.train_seq2seq.train_bart --config configs/train/bart/base/medical_ft.yaml
python -m src.train_seq2seq.train_bart --config configs/train/bart/base/balanced_multidomain_ft.yaml
