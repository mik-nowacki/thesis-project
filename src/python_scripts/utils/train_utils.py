import torch
import numpy as np
import wandb
import optuna
import gc
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

import xgboost as xgb
from wandb.integration.xgboost import WandbCallback
from optuna.integration import XGBoostPruningCallback


def run_xgboost_training(
    params, ds_train, ds_val, Y_val, 
    num_boost_round=600, early_stopping_rounds=20, 
    trial=None, save_dir=None
):
    """
    A universal XGBoost training handler for both standard training and Optuna tuning.
    - trial: If provided, enables Optuna pruning and saves the checkpoint.
    """
    evals_list = [(ds_train, 'train'), (ds_val, 'val')]
    
    # Base callbacks
    callbacks = [WandbCallback()]
    
    # Add Optuna pruning if it's a tuning run
    if trial:
        callbacks.append(XGBoostPruningCallback(trial, "val-rmse"))
        
    try:
        # --- TRAINING ---
        xgb_model = xgb.train(
            params=params, 
            dtrain=ds_train, 
            num_boost_round=num_boost_round, 
            evals=evals_list,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=0, 
            callbacks=callbacks 
        )
        
        # --- EVALUATION ---
        y_pred = xgb_model.predict(ds_val)
        val_mse = mean_squared_error(Y_val, y_pred)
        val_rmse = root_mean_squared_error(Y_val, y_pred)
        val_r2 = r2_score(Y_val, y_pred)
        
        wandb.log({
            "val_mse": val_mse, 
            "val_rmse": val_rmse,
            "val_r2": val_r2
        })
        
        # --- OPTUNA & CHECKPOINT LOGIC ---
        if trial and save_dir:
            xgb_model.save_model(f'{save_dir}/trial_{trial.number}_best.ubj')
            trial.set_user_attr('best_iteration', xgb_model.best_iteration)
            
        return val_rmse

    except RuntimeError as e:
        # Catch PyTorch/CUDA GPU Out of Memory errors
        if "out of memory" in str(e).lower() and trial:
            print(f"\n[Warning] Trial {trial.number} failed due to GPU OOM. Pruning...")
            wandb.log({"error": "GPU Out of Memory"})
            raise optuna.TrialPruned()
        else:
            raise e

    finally:
        # Strict memory cleanup
        if 'xgb_model' in locals():
            del xgb_model
        if 'y_pred' in locals():
            del y_pred
        gc.collect()


def run_pytorch_training(
    model, train_loader, val_loader, optimizer, criterion, device,
    epochs, has_context, model_type, scheduler= None,
    trial=None, save_dir=None, early_stopping_rounds=10
):
    """
    A universal PyTorch training loop for both standard training and Optuna tuning.
    - model_type: 'lstm' or 'itransformer' to handle their specific forward passes.
    - trial: If an Optuna trial is passed, it enables early stopping and pruning.
    """
    best_val_rmse = float('inf')
    early_stopping_counter = 0

    try:
        for epoch in range(epochs):
            # ==============================
            #           TRAINING
            # ==============================
            model.train()
            train_sse = 0.0
            train_preds, train_targets = [], []

            for batch in train_loader:
                # Unpack batch
                if has_context:
                    batch_X, batch_C, batch_Y = batch
                    batch_C = batch_C.to(device)
                else:
                    batch_X, batch_Y = batch
                    batch_C = None
                
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

                optimizer.zero_grad()

                # Handle model-specific forward passes
                if model_type == 'lstm':
                    if has_context:
                        # LSTM expects context concatenated to the feature dimension
                        batch_X = torch.cat([batch_X, batch_C], dim=-1)
                    y_pred = model(batch_X)
                
                elif model_type == 'itransformer':
                    # iTransformer uses a dedicated kwarg for context marks
                    y_pred = model(x_enc=batch_X, x_mark_enc=batch_C)

                elif model_type == 'transformer':
                    #Transformer uses separate kwargs for sequential and static inputs
                    y_pred = model(x_seq=batch_X, x_static=batch_C)

                loss = criterion(y_pred, batch_Y)
                loss.backward()
                optimizer.step()

                train_sse += loss.item() * batch_X.size(0)
                
                # Detach and move to CPU for metric calculation later
                train_preds.append(y_pred.detach().cpu())
                train_targets.append(batch_Y.detach().cpu())

            # Calculate Training Metrics
            train_mse = train_sse / len(train_loader.dataset)
            train_rmse = np.sqrt(train_mse)
            t_preds = torch.cat(train_preds).numpy().flatten()
            t_targs = torch.cat(train_targets).numpy().flatten()
            train_r2 = r2_score(t_targs, t_preds)

            # ==============================
            #          VALIDATION
            # ==============================
            model.eval()
            val_sse = 0.0
            val_preds, val_targets = [], []

            with torch.no_grad():
                for batch in val_loader:
                    if has_context:
                        batch_X, batch_C, batch_Y = batch
                        batch_C = batch_C.to(device)
                    else:
                        batch_X, batch_Y = batch
                        batch_C = None

                    batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

                    if model_type == 'lstm':
                        if has_context:
                            batch_X = torch.cat([batch_X, batch_C], dim=-1)
                        y_pred = model(batch_X)
                    elif model_type == 'itransformer':
                        y_pred = model(x_enc=batch_X, x_mark_enc=batch_C)
                    elif model_type == 'transformer':
                        y_pred = model(x_seq=batch_X, x_static=batch_C)

                    loss = criterion(y_pred, batch_Y)
                    val_sse += loss.item() * batch_X.size(0)
                    
                    val_preds.append(y_pred.cpu())
                    val_targets.append(batch_Y.cpu())
            
            # Calculate Validation Metrics
            val_mse = val_sse / len(val_loader.dataset)
            val_rmse = np.sqrt(val_mse)
            v_preds = torch.cat(val_preds).numpy().flatten()
            v_targs = torch.cat(val_targets).numpy().flatten()
            val_r2 = r2_score(v_targs, v_preds)

            if scheduler is not None:
                scheduler.step()

            # W&B Logging
            wandb.log({
                "train_mse": train_mse, "train_rmse": train_rmse, "train_r2": train_r2,
                "val_mse": val_mse, "val_rmse": val_rmse, "val_r2": val_r2,
                "learning_rate": optimizer.param_groups[0]['lr'], "epoch": epoch
            })

            # ==============================
            #   OPTUNA & CHECKPOINT LOGIC
            # ==============================
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                
                # If Optuna is running, save the best checkpoint for this trial
                if trial and save_dir:
                    early_stopping_counter = 0
                    trial.set_user_attr('best_epoch', epoch)
                    torch.save(model.state_dict(), f'{save_dir}/trial_{trial.number}_best.pt')
            else:
                early_stopping_counter += 1

            if trial:
                trial.report(val_rmse, epoch)
                if trial.should_prune():
                    print(f"Trial {trial.number} pruned at epoch {epoch}.")
                    raise optuna.TrialPruned()
                
                if early_stopping_counter >= early_stopping_rounds:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
        
        return best_val_rmse

    except RuntimeError as e:
        # Gracefully handle GPU OOM errors
        if "out of memory" in str(e).lower() and trial:
            print(f"\n[Warning] Trial {trial.number} failed due to GPU OOM. Pruning...")
            wandb.log({"error": "GPU Out of Memory"})
            raise optuna.TrialPruned()
        else:
            raise e
            
    finally:
        if 'model' in locals():
            del model
        if 'scheduler' in locals():
            del scheduler
        if 'optimizer' in locals():
            del optimizer
        if 'criterion' in locals():
            del criterion

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        gc.collect()