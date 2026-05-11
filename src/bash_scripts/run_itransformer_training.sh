#!/bin/bash
#SBATCH --job-name=eeg-iTrans     
#SBATCH --output=iTrans-%j.out      
#SBATCH --error=iTrans-%j.err   
#SBATCH -p long
#SBATCH --gres=gpu:L4:1           
#SBATCH --cpus-per-task=4         
#SBATCH --mem=32G                 

# read the cmd argument
SEQ_LEN=${1:-200}

# clear the cmd input (so that the conda env is loaded correctly)
set -- 

# Load conda environment
source ~/miniforge3/bin/activate
conda activate thesis_project

# Set paths
cd /data/users/$USER/thesis-project/
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Load Secrets securely from the .env file
# This reads the file and exports the variables into the bash environment
if [ -f /data/users/$USER/thesis-project/.env ]; then
    export $(grep -v '^#' /data/users/$USER/thesis-project/.env | xargs)
else
    echo "Warning: .env file not found!"
fi
# Run the script
echo "Starting iTransformer on GPU for SEQ_LEN: $SEQ_LEN..."
python src/python_scripts/training/train_itransformer.py --seq_len $SEQ_LEN
echo "Job finished."