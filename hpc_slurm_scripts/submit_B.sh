#!/bin/bash
#SBATCH --job-name=cons_B
#SBATCH --output=cons_B_%j.log
#SBATCH --error=cons_B_%j.err
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

source activate llama70b_env
python run_instruct_consensus_B.py
