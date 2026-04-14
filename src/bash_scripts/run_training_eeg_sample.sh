#!/bin/bash

#SBATCH -J train
#SBATCH --gres=gpu

cd /data/users/nowacki/thesis-project/

export PYTHONPATH="$(pwd):$PYTHONPATH"

# Run the script
python src/python_scripts/train_eeg_sample.py
