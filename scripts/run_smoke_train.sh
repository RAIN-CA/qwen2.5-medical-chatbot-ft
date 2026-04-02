#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."
export PYTHONPATH=.

python -m src.train.train_lora --config configs/train/smoke_test.yaml
