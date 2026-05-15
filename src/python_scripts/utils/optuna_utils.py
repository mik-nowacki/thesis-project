import os
import json

def process_optuna_results(study, save_dir, seq_len, has_context, file_extension):
    """
    Prints study results, saves metadata to JSON, and deletes suboptimal trial checkpoints.
    
    Args:
        study: The completed Optuna study object
        save_dir: Directory where checkpoints and JSON will be saved
        seq_len: Sequence length used (for metadata)
        has_context: Boolean for patient context (for metadata)
        file_extension: '.pt' for PyTorch or '.ubj' for XGBoost
    """
    best_trial = study.best_trial

    print("\n--- OPTIMIZATION FINISHED ---")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best RMSE: {best_trial.value:.4f}")
    print("Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Save study metadata
    study_results = {
        'best_trial': best_trial.number,
        'best_val_rmse': best_trial.value,
        'user_attrs': best_trial.user_attrs, # Automatically grabs best_iteration OR best_epoch
        'params': best_trial.params,
        'seq_len': seq_len,
        'p_context': has_context,
    }
    
    with open(os.path.join(save_dir, 'study_results.json'), 'w') as f:
        json.dump(study_results, f, indent=2)

    # Clean up former best trials
    for trial in study.trials:
        path = os.path.join(save_dir, f'trial_{trial.number}_best{file_extension}')
        if trial.number != best_trial.number and os.path.exists(path):
            os.remove(path)