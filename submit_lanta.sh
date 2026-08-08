#!/bin/bash
#SBATCH --job-name=tts-finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=tts_finetune_%j.log
#SBATCH -A xxxxxx

echo "This template for train in Lanta HPC"

# --- Hugging Face Offline Cache Setup ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="${SCRIPT_DIR}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "================================================================="
echo "  [HF Cache Path]     HF_HOME = ${HF_HOME}"
echo "  [HF Offline Mode]   HF_HUB_OFFLINE = 1 (Compute Node Offline)"
echo "================================================================="

echo "load Mamba Module and cudatoolkit"
module load Mamba/23.11.0-0
module load cudatoolkit/24.11_12.6

echo "Activate Conda Environment"
conda activate xxxx

jupyter lab --ip 0.0.0.0 --no-browser
