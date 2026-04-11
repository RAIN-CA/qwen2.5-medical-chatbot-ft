#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."
mkdir -p artifacts

zip -r artifacts/experiments_eval_records.zip outputs/experiments \
  -i "*/predictions/*.jsonl" \
  -i "*/reports/*.json" \
  -i "*.log" \
  -i "*.csv"

echo "Saved: artifacts/experiments_eval_records.zip"
