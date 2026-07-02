#!/bin/bash -l
#SBATCH --job-name=pair-wise-judge
#SBATCH --partition=medium
#SBATCH --mem=32G
#SBATCH --cpus-per-task=3
#SBATCH --time=24:00:00

base_model_name=Qwen3.6-35B-A3B
approach_type=easyep

chmod +x /path/to/moe-pruning-reliability/evaluation/scripts/pruned/pair-wise_llm-as-a-judge_multinews.sh

judge_types=(
    "gemini-3.1"
    "gpt-5.4"
    "claude-4.5"
)
seed_strs=(
    "0"
    "1"
    "2"
)

# Dual 0.5
for seed_str in "${seed_strs[@]}"; do
    for judge_type in "${judge_types[@]}"; do
        approach_name=${base_model_name}-dual-${approach_type}-${seed_str}
        apptainer exec \
            --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/ssl/certs/ca-certificates.crt \
            --bind /etc/pki/ca-trust:/etc/pki/ca-trust \
            ~/containers/vllm-openai_v0.19.1.sif \
            /path/to/moe-pruning-reliability/evaluation/scripts/pruned/pair-wise_llm-as-a-judge_multinews.sh $base_model_name $approach_name $judge_type
    done
    sleep 30
done

# Dual 0.75
for seed_str in "${seed_strs[@]}"; do
    for judge_type in "${judge_types[@]}"; do
        approach_name=${base_model_name}-dual-${approach_type}-${seed_str}-0.75
        apptainer exec \
            --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/ssl/certs/ca-certificates.crt \
            --bind /etc/pki/ca-trust:/etc/pki/ca-trust \
            ~/containers/vllm-openai_v0.19.1.sif \
            /path/to/moe-pruning-reliability/evaluation/scripts/pruned/pair-wise_llm-as-a-judge_multinews.sh $base_model_name $approach_name $judge_type
    done
    sleep 30
done

# GPTQ
for seed_str in "${seed_strs[@]}"; do
    for judge_type in "${judge_types[@]}"; do
        approach_name=${base_model_name}-medinst-${approach_type}-${seed_str}-gptq
        apptainer exec \
            --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/ssl/certs/ca-certificates.crt \
            --bind /etc/pki/ca-trust:/etc/pki/ca-trust \
            ~/containers/vllm-openai_v0.19.1.sif \
            /path/to/moe-pruning-reliability/evaluation/scripts/pruned/pair-wise_llm-as-a-judge_multinews.sh $base_model_name $approach_name $judge_type
    done
    sleep 30
done
