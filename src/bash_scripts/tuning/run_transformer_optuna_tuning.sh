#!/bin/bash
#SBATCH --job-name=Trans-tun     
#SBATCH --output=tuning/history_transformer/Trans-%j.out      
#SBATCH --error=tuning/history_transformer/Trans-%j.err   
#SBATCH -p long
#SBATCH --gres=gpu:L4:1           # Request 1 NVIDIA L4 GPU
#SBATCH --cpus-per-task=4         # Request 4 CPU cores for the PyTorch DataLoader
#SBATCH --mem=32G                 # Request 32GB of RAM             

# read the cmd argument
SEQ_LEN=${1:-200}
P_CONTEXT_FLAG=${2:-""}  # either "--p_context" or ""

# clear the cmd input (so that the conda env is loaded correctly)
set -- 

# Load conda environment
source ~/Master/DoA_pred_venv/bin/activate

# Set paths
cd /data/users/$USER/Master/thesis-project/
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Load Secrets from the .env file
if [ -f /data/users/$USER/Master/thesis-project/.env ]; then
    set -a   
    source .env
    set +a
else
    echo "Warning: .env file not found!"
fi
# Run the script
echo "Starting Transformer on GPU for SEQ_LEN: $SEQ_LEN | $P_CONTEXT_FLAG"
python src/python_scripts/optuna_tuning/tune_transformer.py --seq_len $SEQ_LEN $P_CONTEXT_FLAG
echo "Job finished."