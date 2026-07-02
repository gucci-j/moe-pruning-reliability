
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

model_abbrev=$(basename "$model_name_or_path")
output_dir=/path/to/moe-pruning-reliability/evaluation/logs_multinews/pruned
mkdir -p "$output_dir"

if [[ "$model_abbrev" == *"gpt-oss-20b"* ]]; then
    max_new_tokens=2048
else
    max_new_tokens=1024
fi

# Activate the virtual environment
source ~/envs/moe-pruning-eval/bin/activate

# Run the evaluation script with vLLM backend
cd /path/to/moe-pruning-reliability/evaluation/src

python generate_multi_news_summaries.py \
    --model ${model_name_or_path} \
    --backend vllm \
    --output-json ${output_dir}/${model_abbrev}/multi_news_predictions.json \
    --max-new-tokens ${max_new_tokens} \
    --max-model-len 8192
