#!/bin/bash -l
#SBATCH --job-name=deepeval_multinews_pruned
#SBATCH --partition=medium
#SBATCH --mem=32G
#SBATCH --cpus-per-task=3
#SBATCH --time=48:00:00

chmod +x /path/to/moe-pruning-reliability/evaluation/scripts/pruned/deepeval_multinews.sh

base_model_name=Qwen3-30B-A3B-Instruct-2507
judge_types=(
    "gemini-3.1"
    "gpt-5.4"
    "claude-4.5"
)

for input_file in /path/to/moe-pruning-reliability/evaluation/logs_multinews/pruned/${base_model_name}-medinst-*/multi_news_predictions.json; do
    for judge_type in "${judge_types[@]}"; do
        apptainer exec \
            --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/ssl/certs/ca-certificates.crt \
            --bind /etc/pki/ca-trust:/etc/pki/ca-trust \
            ~/containers/vllm-openai_v0.19.1.sif \
            /path/to/moe-pruning-reliability/evaluation/scripts/pruned/deepeval_multinews.sh $input_file $judge_type
    done
    sleep 60
done
