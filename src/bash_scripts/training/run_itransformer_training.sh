#!/bin/bash
#SBATCH --job-name=iTrans-trai     
#SBATCH --output=training/history_itransformer/iTrans-%j.out      
#SBATCH --error=training/history_itransformer/iTrans-%j.err   
#SBATCH -p long
#SBATCH --gres=gpu:L4:1           
#SBATCH --cpus-per-task=4         
#SBATCH --mem=32G                

# read the cmd argument
SEQ_LEN=${1:-200}
P_CONTEXT_FLAG=${2:-""}  # either "--p_context" or ""

# clear the cmd input (so that the conda env is loaded correctly)
set -- 

# Load conda environment
source ~/miniforge3/bin/activate
conda activate thesis_project

# Set paths
cd /data/users/$USER/thesis-project/
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Load Secrets from the .env file
if [ -f /data/users/$USER/thesis-project/.env ]; then
    set -a   
    source .env
    set +a
else
    echo "Warning: .env file not found!"
fi

# Run the script
echo "Starting iTransformer on GPU for SEQ_LEN: $SEQ_LEN | $P_CONTEXT_FLAG"
python src/python_scripts/training/train_itransformer.py --seq_len $SEQ_LEN $P_CONTEXT_FLAG
echo "Job finished."
