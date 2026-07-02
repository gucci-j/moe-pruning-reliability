#!/bin/bash -l
#SBATCH --job-name=quantize_qwen3.6
#SBATCH --partition=medium-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=vram80g
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8

seed_num=$1
if [ -z "$seed_num" ]; then
    echo "Usage: $0 <seed_num>"
    exit 1
fi

chmod +x /path/to/moe-pruning-reliability/quantizing/scripts/quantize_model.sh

apptainer exec \
    --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/ssl/certs/ca-certificates.crt \
    --bind /etc/pki/ca-trust:/etc/pki/ca-trust \
    --nv ~/containers/vllm-openai_v0.19.1.sif \
    bash /path/to/moe-pruning-reliability/quantizing/scripts/quantize_model.sh $seed_num gptq easyep
