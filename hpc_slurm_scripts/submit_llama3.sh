#!/bin/bash
#SBATCH --job-name=mhs_llama3
#SBATCH --output=mhs_llama3_%j.log
#SBATCH --error=mhs_llama3_%j.err
#SBATCH --partition=gpunew              # CHANGE THIS if your partition is named differently (e.g., 'gpu-a100' or 'gpu-h200')
#SBATCH --gres=gpu:1                 # Requests 1 GPU. If you specifically need the H200, it might be --gres=gpu:h200:1
#SBATCH --time=02:00:00              # 2 hours is more than enough for ~20 sentences * 20 votes
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Load necessary modules (Uncomment and adjust if your HPC requires module loading for CUDA)
# module load cuda/12.1 

# Activate your environment
source activate llama70b_env

# Run the script
echo "Starting Llama-3-70B inference job..."
python run_llama3_mhs.py
echo "Job completed."
