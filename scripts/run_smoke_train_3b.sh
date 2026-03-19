#!/usr/bin/env bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen25-medchat

cd ~/workplace/qwen2.5-medical-chatbot-ft

python src/train/train_lora.py --config configs/train/smoke_test_3b_qlora.yaml
