#!/bin/bash
#SBATCH --job-name=eeg-lstm       # Job name
#SBATCH --output=lstm-%j.out      # Standard output log (%j = Job ID)
#SBATCH --error=lstm-%j.err       # Standard error log
#SBATCH -p long
#SBATCH --gres=gpu:L4:1           # Request 1 NVIDIA L4 GPU
#SBATCH --cpus-per-task=4         # Request 4 CPU cores for the PyTorch DataLoader
#SBATCH --mem=32G                 # Request 32GB of RAM

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
echo "Starting LSTM on GPU for SEQ_LEN: $SEQ_LEN..."
python src/python_scripts/training/train_lstm.py --seq_len $SEQ_LEN
echo "Job finished."