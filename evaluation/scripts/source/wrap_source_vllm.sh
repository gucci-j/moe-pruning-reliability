#!/bin/bash -l
#SBATCH --job-name=eval_source_vllm
#SBATCH --partition=medium-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=vram80g
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8


chmod +x /path/to/moe-pruning-reliability/evaluation/scripts/source/*.sh

apptainer exec \
    --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/ssl/certs/ca-certificates.crt \
    --bind /etc/pki/ca-trust:/etc/pki/ca-trust \
    --nv ~/containers/vllm-openai_v0.19.1.sif \
    bash /path/to/moe-pruning-reliability/evaluation/scripts/source/medinst32_vllm.sh "$@"

apptainer exec \
    --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/ssl/certs/ca-certificates.crt \
    --bind /etc/pki/ca-trust:/etc/pki/ca-trust \
    --nv ~/containers/vllm-openai_v0.19.1.sif \
    bash /path/to/moe-pruning-reliability/evaluation/scripts/source/general_vllm.sh "$@"

apptainer exec \
    --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/ssl/certs/ca-certificates.crt \
    --bind /etc/pki/ca-trust:/etc/pki/ca-trust \
    --nv ~/containers/vllm-openai_v0.19.1.sif \
    bash /path/to/moe-pruning-reliability/evaluation/scripts/source/medhalt_vllm.sh "$@"

apptainer exec \
    --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/ssl/certs/ca-certificates.crt \
    --bind /etc/pki/ca-trust:/etc/pki/ca-trust \
    --nv ~/containers/vllm-openai_v0.19.1.sif \
    bash /path/to/moe-pruning-reliability/evaluation/scripts/source/rct_vllm.sh "$@"

apptainer exec \
    --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/ssl/certs/ca-certificates.crt \
    --bind /etc/pki/ca-trust:/etc/pki/ca-trust \
    --nv ~/containers/vllm-openai_v0.19.1.sif \
    bash /path/to/moe-pruning-reliability/evaluation/scripts/source/multinews_vllm.sh "$@"

apptainer exec \
    --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/ssl/certs/ca-certificates.crt \
    --bind /etc/pki/ca-trust:/etc/pki/ca-trust \
    --nv ~/containers/vllm-openai_v0.19.1.sif \
    bash /path/to/moe-pruning-reliability/evaluation/scripts/source/multimedqa_vllm.sh "$@"

apptainer exec \
    --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/ssl/certs/ca-certificates.crt \
    --bind /etc/pki/ca-trust:/etc/pki/ca-trust \
    --nv ~/containers/vllm-openai_v0.19.1.sif \
    bash /path/to/moe-pruning-reliability/evaluation/scripts/source/mmlu_vllm.sh "$@"

apptainer exec \
    --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/ssl/certs/ca-certificates.crt \
    --bind /etc/pki/ca-trust:/etc/pki/ca-trust \
    --nv ~/containers/vllm-openai_v0.19.1.sif \
    bash /path/to/moe-pruning-reliability/evaluation/scripts/source/hallu_vllm.sh "$@"
