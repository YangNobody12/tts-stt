#!/usr/bin/env bash

# ==============================================================================
# Dataset Downloader Shell Script (load_dataset.sh)
# Uses `huggingface-cli download --repo-type dataset` to download HF datasets
# directly into the project's local cache folder (./hf_cache/datasets).
#
# Optimized for transfer.lanta.nstda.or.th (LANTA HPC Transfer Node)
# No heavy Python / PyTorch / CUDA runtime required.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- HuggingFace Cache Directory Setup ---
HF_CACHE_DIR="${HF_CACHE_DIR:-${SCRIPT_DIR}/hf_cache}"
DATASETS_CACHE="${HF_CACHE_DIR}/datasets"
mkdir -p "${DATASETS_CACHE}"

export HF_HOME="${HF_CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE_DIR}/hub"
export HF_DATASETS_CACHE="${DATASETS_CACHE}"

echo "================================================================="
echo "  [HF Cache] HF_HOME              = ${HF_HOME}"
echo "  [HF Cache] HF_DATASETS_CACHE    = ${HF_DATASETS_CACHE}"
echo "================================================================="

# Find huggingface-cli executable or fallback to python module
if command -v huggingface-cli &> /dev/null; then
    HF_CMD="huggingface-cli"
elif command -v python3 &> /dev/null && python3 -m huggingface_hub.cli.huggingface_cli --help &> /dev/null; then
    HF_CMD="python3 -m huggingface_hub.cli.huggingface_cli"
elif command -v python &> /dev/null && python -m huggingface_hub.cli.huggingface_cli --help &> /dev/null; then
    HF_CMD="python -m huggingface_hub.cli.huggingface_cli"
else
    echo "[!] Warning: 'huggingface-cli' was not found."
    echo "[*] Attempting to install 'huggingface_hub' via pip..."
    pip install huggingface_hub
    if command -v huggingface-cli &> /dev/null; then
        HF_CMD="huggingface-cli"
    else
        HF_CMD="python3 -m huggingface_hub.cli.huggingface_cli"
    fi
fi

# Default datasets if no argument provided
DEFAULT_DATASETS=(
    "Thanarit/Thai-Voice-Test7"
)

if [ $# -gt 0 ]; then
    DATASETS=("$@")
else
    echo "[*] No dataset specified. Downloading default dataset:"
    for d in "${DEFAULT_DATASETS[@]}"; do
        echo "  - $d"
    done
    DATASETS=("${DEFAULT_DATASETS[@]}")
fi

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "[*] Downloading dataset: ${DATASET} to ${DATASETS_CACHE}..."
    $HF_CMD download --repo-type dataset "${DATASET}" --cache-dir "${DATASETS_CACHE}"
done

echo ""
echo "[✓] All dataset downloads finished! Saved in: ${DATASETS_CACHE}"
