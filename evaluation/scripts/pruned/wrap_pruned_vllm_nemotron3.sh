#!/bin/bash -l
#SBATCH --job-name=eval_pruned_vllm_nemotron3
#SBATCH --partition=medium-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=vram80g
#SBATCH --mem=128GB
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8

nvidia-smi
model_type=NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
chmod +x /path/to/moe-pruning-reliability/evaluation/scripts/pruned/*.sh
model_base_dir=/path/to/models

seed_strs=(
    "0"
    "1"
    "2"
)
pruning_methods=(
    "ean"
    "easyep"
    "frequency"
    "gating"
    "random"
    "reap"
)

for seed_str in "${seed_strs[@]}"; do
    for method in "${pruning_methods[@]}"; do
        echo "Running $method with seed $seed_str"
        model_path="${model_base_dir}/${model_type}-medinst-${method}-${seed_str}"


        apptainer exec \
            --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/ssl/certs/ca-certificates.crt \
            --bind /etc/pki/ca-trust:/etc/pki/ca-trust \
            --nv ~/containers/vllm-openai_v0.19.1.sif \
            bash /path/to/moe-pruning-reliability/evaluation/scripts/pruned/medinst32_vllm.sh ${model_path}

        apptainer exec \
            --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/ssl/certs/ca-certificates.crt \
            --bind /etc/pki/ca-trust:/etc/pki/ca-trust \
            --nv ~/containers/vllm-openai_v0.19.1.sif \
            bash /path/to/moe-pruning-reliability/evaluation/scripts/pruned/general_vllm.sh ${model_path}

        apptainer exec \
            --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/ssl/certs/ca-certificates.crt \
            --bind /etc/pki/ca-trust:/etc/pki/ca-trust \
            --nv ~/containers/vllm-openai_v0.19.1.sif \
            bash /path/to/moe-pruning-reliability/evaluation/scripts/pruned/medhalt_vllm.sh ${model_path}

        apptainer exec \
            --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/ssl/certs/ca-certificates.crt \
            --bind /etc/pki/ca-trust:/etc/pki/ca-trust \
            --nv ~/containers/vllm-openai_v0.19.1.sif \
            bash /path/to/moe-pruning-reliability/evaluation/scripts/pruned/rct_vllm.sh ${model_path}


        apptainer exec \
            --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/ssl/certs/ca-certificates.crt \
            --bind /etc/pki/ca-trust:/etc/pki/ca-trust \
            --nv ~/containers/vllm-openai_v0.19.1.sif \
            bash /path/to/moe-pruning-reliability/evaluation/scripts/pruned/multinews_vllm.sh ${model_path}

        apptainer exec \
            --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/ssl/certs/ca-certificates.crt \
            --bind /etc/pki/ca-trust:/etc/pki/ca-trust \
            --nv ~/containers/vllm-openai_v0.19.1.sif \
            bash /path/to/moe-pruning-reliability/evaluation/scripts/pruned/multimedqa_vllm.sh ${model_path}
        
        apptainer exec \
            --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/ssl/certs/ca-certificates.crt \
            --bind /etc/pki/ca-trust:/etc/pki/ca-trust \
            --nv ~/containers/vllm-openai_v0.19.1.sif \
            bash /path/to/moe-pruning-reliability/evaluation/scripts/pruned/mmlu_vllm.sh ${model_path}

        apptainer exec \
            --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/ssl/certs/ca-certificates.crt \
            --bind /etc/pki/ca-trust:/etc/pki/ca-trust \
            --nv ~/containers/vllm-openai_v0.19.1.sif \
            bash /path/to/moe-pruning-reliability/evaluation/scripts/pruned/hallu_vllm.sh ${model_path}

    done
done