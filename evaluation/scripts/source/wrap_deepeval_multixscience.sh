#!/bin/bash -l
#SBATCH --job-name=deepeval_multixscience_source
#SBATCH --partition=short
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

chmod +x /path/to/moe-pruning-reliability/evaluation/scripts/source/deepeval_multixscience.sh

judge_types=(
    "gemini-3.1"
    "gpt-5.4"
    "claude-4.5"
)

for judge_type in "${judge_types[@]}"; do
    apptainer exec \
        --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/ssl/certs/ca-certificates.crt \
        --bind /etc/pki/ca-trust:/etc/pki/ca-trust \
        ~/containers/vllm-openai_v0.19.1.sif \
        /path/to/moe-pruning-reliability/evaluation/scripts/source/deepeval_multixscience.sh $1 $judge_type
done

