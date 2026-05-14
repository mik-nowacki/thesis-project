#!/bin/bash
#SBATCH --job-name=eeg-xgb     
#SBATCH --output=history_xgb/xgb-%j.out      
#SBATCH --error=history_xgb/xgb-%j.err   
#SBATCH -p long
#SBATCH --gres=gpu:L4:1           
#SBATCH --cpus-per-task=4         
#SBATCH --mem=32G                 

# read the cmd argument
SEQ_LEN=${1:-60}

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

# Run it
echo "Starting XGBoost on GPU for SEQ_LEN: $SEQ_LEN..."
python src/python_scripts/optuna_tuning/tune_xgb.py --seq_len $SEQ_LEN
echo "Job finished."