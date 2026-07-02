#!/bin/bash

# Input arguments
model_name_or_path=$1
if [ -z "$model_name_or_path" ]; then
    echo "Usage: $0 <model_name_or_path>"
    exit 1
fi

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
model_abbrev=$(cut -d'/' -f2 <<< $model_name_or_path)
output_dir=/path/to/moe-pruning-reliability/evaluation/logs_medinst32/source
mkdir -p "$output_dir"

# Set max_new_tokens to 2048 if model_abbrev is gpt-oss-20b
if [[ "$model_abbrev" == "gpt-oss-20b" ]]; then
    max_new_tokens=2048
else
    max_new_tokens=1024
fi

# Activate the virtual environment
source ~/envs/moe-pruning-eval/bin/activate

# Run the evaluation script with vLLM backend
cd /path/to/moe-pruning-reliability/evaluation/src
python3 eval_medinst32.py \
  --name $model_abbrev \
  --dir $output_dir \
  --backend vllm \
  --model $model_name_or_path \
  --max_new_tokens $max_new_tokens \
  --max_model_len 4096 \
  --temperature 0.0
