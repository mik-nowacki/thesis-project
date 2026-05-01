#!/bin/bash
#SBATCH --job-name=eeg-iTrans     
#SBATCH --output=iTrans-%j.out      
#SBATCH --error=iTrans-%j.err       
#SBATCH --gres=gpu:L4:1           
#SBATCH --cpus-per-task=4         
#SBATCH --mem=16G                 

# 1. Load Minerva environment
source ~/miniforge3/bin/activate
conda activate thesis_project

# 2. Set the Python path to the root of the project
cd /data/users/nowacki/thesis-project/
export PYTHONPATH="$(pwd):$PYTHONPATH"

# 3. Authenticate Weights & Biases headless
# Load Secrets securely from the .env file
# This reads the file and exports the variables into the bash environment
if [ -f /data/users/nowacki/thesis-project/.env ]; then
    export $(grep -v '^#' /data/users/nowacki/thesis-project/.env | xargs)
else
    echo "Warning: .env file not found!"
fi
# 4. Run the script
echo "Starting iTransformer on GPU..."
python src/python_scripts/train_itransformer.py
echo "Job finished."