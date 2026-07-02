#!/bin/bash -l
#SBATCH --job-name=all_gpt-oss
#SBATCH --partition=medium-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=vram80g
#SBATCH --mem=128GB
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=12

seed_nums=(
    0
    1
    2
)
pruning_methods=(
    "ean"
    "easyep"
    "frequency"
    "gating"
    "random"
    "reap"
)
ratios=(
    0.125
    0.25
    0.375
    0.625
    0.75
    0.875
)
ratio_to_expert_num() {
    local ratio="$1"
    local total_experts=32
    awk -v ratio="$ratio" -v total="$total_experts" 'BEGIN { printf "%d\n", (1 - ratio) * total }'
}

for seed_num in "${seed_nums[@]}"; do
    for method in "${pruning_methods[@]}"; do
        for ratio in "${ratios[@]}"; do
            echo "Running $method with seed $seed_num and ratio $ratio (experts to keep: $(ratio_to_expert_num $ratio))"
            chmod +x /path/to/moe-pruning-reliability/pruning/scripts/gpt-oss/${method}_bio_ratio.sh

            apptainer exec \
                --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/ssl/certs/ca-certificates.crt \
                --bind /etc/pki/ca-trust:/etc/pki/ca-trust \
                --nv ~/containers/vllm-openai_v0.19.1.sif \
                bash /path/to/moe-pruning-reliability/pruning/scripts/gpt-oss/${method}_bio_ratio.sh $seed_num $ratio $(ratio_to_expert_num $ratio)
        done    
    done
done
