#!/bin/bash
#SBATCH --job-name=eeg-xgb     
#SBATCH --output=xgb-%j.out      
#SBATCH --error=xgb-%j.err   
#SBATCH -p long
#SBATCH --gres=gpu:L4:1           
#SBATCH --cpus-per-task=4         
#SBATCH --mem=32G                 

# Load your Minerva environment
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

# Run it
echo "Starting XGBoost on GPU..."
python src/python_scripts/training/train_xgb.py
echo "Job finished."