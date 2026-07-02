#!/bin/bash

seed_num=$1
if [ -z "$seed_num" ]; then
    echo "Usage: $0 <seed_num> [quant_type: gptq|fp8]"
    exit 1
fi

quant_type=${2:-"gptq"}
pruning_method=${3:-"easyep"}

# Set configurations
export HF_TOKEN=$(cat $HOME/huggingface/token 2>/dev/null || echo "")
export TRANSFORMERS_VERBOSITY=debug
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HOME=/path/to/cache/
export HF_HUB_CACHE=/path/to/cache/
export HF_DATASETS_CACHE=/path/to/cache/
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Adjust paths as needed
PRUNED_MODEL_DIR="/path/to/models/Qwen3.6-35B-A3B-medinst-${pruning_method}-${seed_num}"
OUTPUT_DIR="/path/to/models/Qwen3.6-35B-A3B-medinst-${pruning_method}-${seed_num}-${quant_type}"

mkdir -p $OUTPUT_DIR

# Activate the virtual environment
source ~/envs/moe-pruning/bin/activate

# Run the quantization script
cd /path/to/moe-pruning-reliability/quantizing/src

if [ "$quant_type" == "gptq" ]; then
    echo "Running GPTQ quantization..."
    python3 quantize.py \
        --model_name_or_path "$PRUNED_MODEL_DIR" \
        --quant_type "$quant_type" \
        --dataset_type "medinst" \
        --n_samples 128 \
        --save_path "$OUTPUT_DIR" \
        --data_seed "$seed_num" \
        --sequential_targets "GroupAttention,MoEBlocks"
fi
