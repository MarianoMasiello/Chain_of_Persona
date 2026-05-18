#!/bin/bash
#SBATCH --job-name=persona_gen
#SBATCH --output=persona_gen_%j.log
#SBATCH --error=persona_gen_%j.err
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

source activate llama70b_env

echo "Starting 2-Stage Persona Generation Sweep (T=0.0 to 2.0)..."
python run_persona_sweep.py
echo "Job completed."
