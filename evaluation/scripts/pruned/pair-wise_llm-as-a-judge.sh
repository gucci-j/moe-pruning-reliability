#!/bin/bash

base_model_name=$1
approach_name=$2
judge_type=$3
if [ -z "$base_model_name" ] || [ -z "$approach_name" ] || [ -z "$judge_type" ]; then
    echo "Usage: $0 <base_model_name> <approach_name> <judge_type>"
    exit 1
fi

# Set configurations
export TRANSFORMERS_VERBOSITY=debug
export HF_HOME=/path/to/cache/
export HF_HUB_CACHE=/path/to/cache/
export HF_DATASETS_CACHE=/path/to/cache/
export HF_DATASETS_TRUST_REMOTE_CODE=true
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
hallu_log_dir=/path/to/moe-pruning-reliability/evaluation/logs_hallu
log_base_dir=/path/to/moe-pruning-reliability/evaluation/logs_pair-wise
token_path=/home/kljp789/openai/token
mkdir -p $log_base_dir
output_dir=$log_base_dir/$base_model_name
mkdir -p $output_dir

# Activate the evaluation environment
source ~/envs/moe-pruning-eval/bin/activate

# Run the evaluation script
cd /path/to/moe-pruning-reliability/evaluation/src

if [ "$judge_type" == "gemini-3.1" ]; then
    model="google/gemini-3.1-flash-lite-preview"
    gateway_endpoint="vertex-ai-openai"
elif [ "$judge_type" == "gpt-5.4" ]; then
    model="gpt-5.4-mini"
    gateway_endpoint="azure-openai"
elif [ "$judge_type" == "claude-4.5" ]; then
    model="us.anthropic.claude-haiku-4-5-20251001-v1:0"
    gateway_endpoint="bedrock"
else
    echo "Unsupported judge type: $judge_type"
    exit 1
fi

python eval_pair-wise_llm-as-a-judge.py \
    --gateway-endpoint $gateway_endpoint \
    --model $model \
    --vanilla-log $hallu_log_dir/source/$base_model_name.json \
    --pruned-log $hallu_log_dir/pruned/210426/$approach_name.json \
    --output-dir $output_dir \
    --max-completion-tokens 4096 \
    --log-every 10 \
    --token-path $token_path \
    --num-workers 2
