#!/bin/bash -l
#SBATCH --job-name=easyep_bio_gpt_oss
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=vram80g
#SBATCH --mem=128GB
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8

seed_num=$1
if [ -z "$seed_num" ]; then
    echo "Usage: $0 <seed_num>"
    exit 1
fi

chmod +x /path/to/moe-pruning-reliability/pruning/scripts/gpt-oss/easyep_bio.sh

apptainer exec \
    --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/ssl/certs/ca-certificates.crt \
    --bind /etc/pki/ca-trust:/etc/pki/ca-trust \
    --nv ~/containers/vllm-openai_v0.19.1.sif \
    bash /path/to/moe-pruning-reliability/pruning/scripts/gpt-oss/easyep_bio.sh $seed_num
