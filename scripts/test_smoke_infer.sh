#!/usr/bin/env bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen25-medchat

cd ~/workplace/qwen2.5-medical-chatbot-ft

python src/inference/chat_infer.py \
  --adapter_path outputs/qwen25_medchat_smoke \
  --query "What are the common symptoms of diabetes?"
