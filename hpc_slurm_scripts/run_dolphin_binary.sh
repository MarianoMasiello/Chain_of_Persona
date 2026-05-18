#!/bin/bash
#SBATCH --job-name=dolphin_bin
#SBATCH --partition=gpuh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=dolphin_binary_%j.log
#SBATCH --error=dolphin_binary_%j.err

# Initialize conda for the script 
eval "$(conda shell.bash hook)"

# Activate the environment the right way
conda activate llama70b_env

echo "Starting Dolphin Binary 2-Stage Sweep..."
python run_dolphin_binary.py
echo "Job completed."
