#!/bin/bash

#SBATCH --job-name=pythia_train    # Job name
#SBATCH --output=slurm_logs/pythia_train_%j.out # Standard output log (%j expands to job ID)
#SBATCH --error=slurm_logs/pythia_train_%j.err  # Standard error log (%j expands to job ID)
#SBATCH --partition=long          # Partition name (e.g., gpu, mila, long - CHECK YOUR CLUSTER'S CONFIG)
#SBATCH --time=24:00:00             # Time limit hrs:min:sec (adjust based on expected runtime)
#SBATCH --cpus-per-task=8           # Number of CPU cores per task (adjust based on dataloader workers, etc.)
#SBATCH --mem=12G                   # Memory per node (e.g., 64G, 128G - adjust based on model/data size)
#SBATCH --gres=gpu:1                # Number of GPUs per node (adjust if you need more/less)
#SBATCH --constraint=ampere

# --- Safety measures ---
set -euo pipefail # Fail fast on errors, unset variables, and pipe errors

# --- Environment Setup ---
echo "Setting up environment..."
# Load necessary modules (example, CHECK YOUR CLUSTER'S MODULES)
module purge # Start with a clean environment
module load python/3.10 #cuda/11.8 cudnn/8.6 # Adjust versions as needed

# Activate your virtual environment
VENV_PATH="/network/scratch/h/howeniko/el/venv" # <-- CHANGE THIS TO YOUR VENV PATH
if [ -d "$VENV_PATH" ]; then
    echo "Activating virtual environment: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
else
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Create log directory if it doesn't exist
mkdir -p slurm_logs

# Navigate to the directory containing the script (optional, assumes submission from workspace root)
cd /network/scratch/h/howeniko/el # Or use $SLURM_SUBMIT_DIR if submitting from the script's dir

# --- Job Execution ---
echo "Starting Python script: pretrain_model.py"

# Run the training script
# The environment variables for caching (HF_DATASETS_CACHE) and parallelism are now set inside train_model.py
python pretrain_model.py

# --- Job Completion ---
echo "Python script finished."
deactivate # Deactivate virtual environment
echo "Job completed successfully."

