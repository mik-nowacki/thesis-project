#!/bin/bash
#SBATCH --job-name=eeg-lstm       # Job name
#SBATCH --output=lstm-%j.out      # Standard output log (%j = Job ID)
#SBATCH --error=lstm-%j.err       # Standard error log
#SBATCH --gres=gpu:L4:1           # Request 1 NVIDIA L4 GPU[cite: 1, 2]
#SBATCH --cpus-per-task=4         # Request 4 CPU cores for the PyTorch DataLoader[cite: 1]
#SBATCH --mem=16G                 # Request 16GB of RAM[cite: 1]

source ~/miniforge3/bin/activate
conda activate thesis_project

cd /data/users/nowacki/thesis-project/

export PYTHONPATH="$(pwd):$PYTHONPATH"

# Load Secrets securely from the .env file
# This reads the file and exports the variables into the bash environment
if [ -f /data/users/nowacki/thesis-project/.env ]; then
    export $(grep -v '^#' /data/users/nowacki/thesis-project/.env | xargs)
else
    echo "Warning: .env file not found!"
fi


# 3. Run the script
echo "Starting training job on GPU..."
python src/python_scripts/training/train_lstm.py
echo "Job finished."