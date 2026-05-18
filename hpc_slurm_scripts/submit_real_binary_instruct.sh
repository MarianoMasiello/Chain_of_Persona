#!/bin/bash
#SBATCH --job-name=binary_instruct
#SBATCH --output=binary_instruct_%j.log
#SBATCH --error=binary_instruct_%j.err
#SBATCH --partition=gpunew
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

source activate llama70b_env

echo "Starting Unified Binary Instruct Run (Baseline + Grid)..."
python run_original_binary.py
echo "Job completed."
