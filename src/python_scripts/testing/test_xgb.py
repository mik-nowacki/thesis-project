
# # Evaluate the best iteration
# Y_test_flat = Y_val_clean.flatten() # match predictions shape ( (367597) instead of (367597, 1))
# predictions = xgb_model.predict(ds_val)
# mae = mean_absolute_error(Y_test_flat, predictions)
# mse = mean_squared_error(Y_test_flat, predictions)
# rmse = root_mean_squared_error(Y_test_flat, predictions)
# r2 = r2_score(Y_test_flat, predictions)

# # custom tolerance accurarcy metric
# abs_errors = np.abs(Y_test_flat - predictions)
# within_tolerance_25 = abs_errors <= 2.5
# tolerance_accuracy_25 = np.mean(within_tolerance_25) * 100

# within_tolerance_5 = abs_errors <= 5.0
# tolerance_accuracy_5 = np.mean(within_tolerance_5) * 100

# within_tolerance_10 = abs_errors <= 10.0
# tolerance_accuracy_10 = np.mean(within_tolerance_10) * 100


# # Log final metrics for this trial
# wandb.log({
#             "test_mae": mae, 
#             "test_mse": mse, 
#             "test_rmse": rmse,
#             "test_r2_score": r2,
#             "test_tolerance_accuracy_2_5": tolerance_accuracy_25,
#             "test_tolerance_accuracy_5": tolerance_accuracy_5,
#             "test_tolerance_accuracy_10": tolerance_accuracy_10,
#         })