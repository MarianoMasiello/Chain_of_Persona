#!/bin/bash
#SBATCH --job-name=2stage_full
#SBATCH --output=2stage_full_%j.log
#SBATCH --error=2stage_full_%j.err
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --time=22:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

source activate llama70b_env

echo "Starting 2-Stage Pipeline COMPLETE Grid Search (Remaining 72 combinations)..."
python run_2stage_full_grid.py
echo "Job completed."
