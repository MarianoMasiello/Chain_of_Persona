#!/bin/bash
#SBATCH --job-name=bin_instruct
#SBATCH --output=bin_instruct_%j.log
#SBATCH --error=bin_instruct_%j.err
#SBATCH --partition=gpunew
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

source activate llama70b_env

echo "Starting Sandwiched Binary Instruct Run (Baseline + Grid)..."
python run_binary_instruct_sandwiched.py
echo "Job completed."
