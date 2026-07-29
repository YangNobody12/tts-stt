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

echo "load Mamba Module and cudatoolkit"
module load Mamba/23.11.0-0
module load cudatoolkit/24.11_12.6

echo "Activate Conda Environment"
conda activate xxxx

jupyter lab --ip 0.0.0.0 --no-browser
