#!/bin/bash

# Input arguments
model_name_or_path=$1
# Set configurations
export TRANSFORMERS_VERBOSITY=debug
export HF_HOME=/path/to/cache/
export HF_HUB_CACHE=/path/to/cache/
export HF_DATASETS_CACHE=/path/to/cache/
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_ALLOW_CODE_EVAL="1"
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export EXTERNAL_PATH=/path/to/moe-pruning-reliability/external
model_abbrev=$(basename "$model_name_or_path")
output_dir=/path/to/moe-pruning-reliability/evaluation/logs_medinst32/pruned
mkdir -p "$output_dir"
max_new_tokens=1024

# Activate the virtual environment
source ~/envs/moe-pruning-eval/bin/activate

# Run the evaluation script with vLLM backend
cd /path/to/moe-pruning-reliability/evaluation/src

python eval_medinst32.py \
  --name $model_abbrev \
  --dir $output_dir \
  --backend vllm \
  --model $model_name_or_path \
  --max_new_tokens $max_new_tokens \
  --temperature 0.0 \
  --max_model_len 4096
