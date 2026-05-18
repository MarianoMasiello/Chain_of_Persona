#!/bin/bash
#SBATCH --job-name=binary_base
#SBATCH --output=binary_base_%j.log
#SBATCH --error=binary_base_%j.err
#SBATCH --partition=gpunew
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

source activate llama70b_env

echo "Ensuring bitsandbytes is installed for the BNB Base Model..."
pip install -q bitsandbytes accelerate

echo "Starting BASE Model Binary Sweep (Baseline + Grid)..."
python run_binary_base.py
echo "Job completed."
