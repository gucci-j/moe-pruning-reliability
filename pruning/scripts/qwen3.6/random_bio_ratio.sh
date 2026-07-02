#!/bin/bash

seed_num=$1
ratio_num=$2
n_experts_to_keep=$3

if [ -z "$seed_num" ] || [ -z "$ratio_num" ] || [ -z "$n_experts_to_keep" ]; then
    echo "Usage: $0 <seed_num> <ratio_num> <n_experts_to_keep>"
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

output_dir=/path/to/models/Qwen3.6-35B-A3B-medinst-random-${seed_num}-${ratio_num}
mkdir -p $output_dir

# Activate the virtual environment
source ~/envs/moe-pruning/bin/activate

# Run the training script
cd /path/to/moe-pruning-reliability/pruning/src
python3 prune.py \
    --model_name_or_path "Qwen/Qwen3.6-35B-A3B" \
    --dataset_type "medinst" \
    --n_samples 128 \
    --n_experts_to_keep $n_experts_to_keep \
    --save_path "$output_dir" \
    --pruning_method "random" \
    --pruning_type "physical" \
    --verbose \
    --data_seed $seed_num
