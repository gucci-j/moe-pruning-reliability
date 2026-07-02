
#!/bin/bash

# Input arguments
model_name_or_path=$1
# Set configurations
export TRANSFORMERS_VERBOSITY=debug
export HF_HOME=/path/to/cache/
export HF_HUB_CACHE=/path/to/cache/
export HF_DATASETS_CACHE=/path/to/cache/
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_ALLOW_CODE_EVAL="1"
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export EXTERNAL_PATH=/path/to/moe-pruning-reliability/external
model_abbrev=$(basename "$model_name_or_path")
rct_dir=/path/to/moe-pruning-reliability/evaluation/logs_rct/pruned
medinst_dir=/path/to/moe-pruning-reliability/evaluation/logs_medinst32/pruned
output_dir=/path/to/moe-pruning-reliability/evaluation/logs_hallu/pruned
mkdir -p "$output_dir"

# Activate the virtual environment
source ~/envs/moe-pruning-eval/bin/activate

# Run the evaluation script with vLLM backend
cd /path/to/moe-pruning-reliability/evaluation/src

python -c "import nltk; nltk.download('punkt_tab')"

python eval_hallu.py \
    --rct-json ${rct_dir}/${model_abbrev}/rct_summaries_predictions.json \
    --medinst-json ${medinst_dir}/${model_abbrev}.json \
    --medinst-reference-mode input \
    --medinst-config-suffix "" \
    --output-json ${output_dir}/${model_abbrev}.json \
    --output-records-json ${output_dir}/${model_abbrev}_records.json
