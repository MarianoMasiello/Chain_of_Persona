#!/bin/bash
#SBATCH --job-name=base_cons
#SBATCH --output=base_cons_%j.log
#SBATCH --error=base_cons_%j.err
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

source activate llama70b_env

echo "Ensuring bitsandbytes is installed..."
pip install -q bitsandbytes accelerate

echo "Starting Base Model Consensus Baseline..."
python run_base_consensus.py
echo "Job completed."
