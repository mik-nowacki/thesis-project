#!/bin/bash
#SBATCH --job-name=xgb-trai     
#SBATCH --output=training/history_xgb/xgb-%j.out      
#SBATCH --error=training/history_xgb/xgb-%j.err  
 
#SBATCH --gres=gpu:L4:1           # Request 1 NVIDIA L4 GPU
#SBATCH --cpus-per-task=4         # Request 4 CPU cores for the PyTorch DataLoader
#SBATCH --mem=32G                 # Request 32GB of RAM

# read the cmd argument
SEQ_LEN=${1:-60}
P_CONTEXT_FLAG=${2:-""}  # either "--p_context" or ""

# clear the cmd input (so that the conda env is loaded correctly)
set -- 

# Load conda environment
source ~/miniforge3/bin/activate
conda activate thesis_project

# Set paths
cd /data/users/$USER/thesis-project/
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Load Secrets securely from the .env file
if [ -f /data/users/$USER/thesis-project/.env ]; then
    set -a   
    source .env
    set +a
else
    echo "Warning: .env file not found!"
fi

# Run the script
echo "Starting XGBoost on GPU for SEQ_LEN: $SEQ_LEN | $P_CONTEXT_FLAG"
python src/python_scripts/training/train_xgb.py --seq_len $SEQ_LEN $P_CONTEXT_FLAG
echo "Job finished."