#!/bin/bash

# Arguments
model_name=$1
if [ -z "$model_name" ]; then
    echo "Usage: $0 <model_name>"
    exit 1
fi

# Configs

export TRANSFORMERS_VERBOSITY=debug
export HF_HOME=/path/to/cache/
export HF_HUB_CACHE=/path/to/cache/
export HF_DATASETS_CACHE=/path/to/cache/
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_ALLOW_CODE_EVAL="1"
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export EVAL_DATASET_PATH=/path/to/cache/datasets
export EXTERNAL_PATH=/path/to/moe-pruning-reliability/external
log_base_dir="/path/to/moe-pruning-reliability/evaluation/logs_lmeval/source"

# Activate environment
source ~/envs/moe-pruning-eval/bin/activate

# Change model_args as needed
if [[ $model_name == "openai/gpt-oss-20b" ]]; then
    model_args="{\"pretrained\":\"openai/gpt-oss-20b\",\"dtype\":\"bfloat16\",\"chat_template_args\":{\"reasoning_effort\":\"low\"},\"enable_thinking\":true,\"think_end_token\":200008,\"trust_remote_code\":true,\"gpu_memory_utilization\":0.9,\"max_model_len\":4096}"
else
    model_args="pretrained=${model_name},dtype=bfloat16,enable_thinking=False,trust_remote_code=True,gpu_memory_utilization=0.9,max_model_len=4096"
fi

# Run evaluation
lm-eval --model vllm \
    --model_args=${model_args} \
    --tasks=leaderboard_ifeval \
    --batch_size auto \
    --output_path="${log_base_dir}/general" \
    --num_fewshot 0 \
    --apply_chat_template \
    --fewshot_as_multiturn

lm-eval --model vllm \
    --model_args=${model_args} \
    --tasks=gsm8k \
    --batch_size auto \
    --output_path="${log_base_dir}/general" \
    --num_fewshot 5 \
    --apply_chat_template \
    --fewshot_as_multiturn \

lm-eval --model vllm \
    --model_args=${model_args} \
    --tasks=humaneval_instruct \
    --batch_size auto \
    --output_path="${log_base_dir}/general" \
    --num_fewshot 0 \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --confirm_run_unsafe_code
