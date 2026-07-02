
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
export HF_HUB_ETAG_TIMEOUT=86400
export HF_HUB_DOWNLOAD_TIMEOUT=86400
export VLLM_WORKER_MULTIPROC_METHOD=spawn
batch_size=-1
model_abbrev=$(basename "$model_name_or_path")
output_dir=/path/to/moe-pruning-reliability/evaluation/logs_mmlu/pruned/$model_abbrev
mkdir -p "$output_dir"

# Set max_new_tokens to 1024 if model_abbrev contains gpt-oss-20b
if [[ "$model_abbrev" == *"gpt-oss-20b"* ]]; then
    max_new_tokens=1024
else
    max_new_tokens=128
fi

# Activate the virtual environment
source ~/envs/moe-pruning-eval/bin/activate

# Run the evaluation script with vLLM backend
cd /path/to/moe-pruning-reliability/evaluation/src

python3 eval_mmlu.py \
    --name ${model_abbrev} \
    --output-dir ${output_dir} \
    --backend vllm \
    --model ${model_name_or_path} \
    --tasks all \
    --batch-size ${batch_size} \
    --max-new-tokens ${max_new_tokens} \
    --max-model-len 4096
