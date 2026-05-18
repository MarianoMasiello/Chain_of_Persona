#!/bin/bash
#SBATCH --job-name=cons_A
#SBATCH --output=cons_A_%j.log
#SBATCH --error=cons_A_%j.err
#SBATCH --partition=gpunew
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

source activate llama70b_env
python run_instruct_consensus_A.py
