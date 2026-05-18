#!/bin/bash
#SBATCH --job-name=mhs_ablation
#SBATCH --output=mhs_ablation_%j.log
#SBATCH --error=mhs_ablation_%j.err
#SBATCH --partition=gpunew
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

source activate llama70b_env

echo "Starting Temperature Ablation Study..."
python run_ablation.py
echo "Job completed."
