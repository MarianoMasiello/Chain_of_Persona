#!/bin/bash
#SBATCH --account=3200991
#SBATCH --job-name=llama70b_test
#SBATCH --partition=gpuh200        # Use the appropriate partition for your node
#SBATCH --time=01:00:00

# Compute Resources
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8           # 70B needs a bit more CPU power to load
#SBATCH --mem=128G                   # Request 64GB of CPU RAM

# Request exactly 1 NVIDIA H200 GPU
#SBATCH --gres=gpu:H200:1

# Standard output and error logs
#SBATCH --output=slurm_70b_test_%j.out
#SBATCH --error=slurm_70b_test_%j.err

echo "Llama-3 70B Test started on $(hostname)"

# Activate Conda
module load /software/modules/miniconda3
eval "$(conda shell.bash hook)"
conda activate llama70b_env

export PYTHONUNBUFFERED=1

# Tell the C++ compiler to chill out and only use 4 workers
export MAX_JOBS=4

# Run the Python Script
python run_70b.py

echo "Llama-3 70B Test Complete!"
