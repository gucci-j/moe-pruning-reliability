#!/bin/bash

input_file=$1
judge_type=$2
if [ -z "$input_file" ] || [ -z "$judge_type" ]; then
    echo "Usage: $0 <input_file> <judge_type>"
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
log_base_dir=/path/to/moe-pruning-reliability/evaluation/logs_deepeval/rct/source
mkdir -p $log_base_dir
model_name=$(basename $(dirname $input_file))
output_dir=$log_base_dir/$model_name
mkdir -p $output_dir

# Activate the evaluation environment
source ~/envs/moe-pruning-eval/bin/activate

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

# Run the evaluation script
cd /path/to/moe-pruning-reliability/evaluation/src
python deepeval_rct.py \
  --input $input_file \
  --output-dir $output_dir \
  --model $model \
  --gateway-endpoint $gateway_endpoint \
  --threshold 0.5 \
  --num-workers 2
