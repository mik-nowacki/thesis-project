# inference.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from ForecastingModel import ForecastingModel

# 1. Configuration (Must match training exactly)
seq_len = 200
device = 'cpu'
FORECAST_STEPS = 200

# 2. Initialize the architecture
model = ForecastingModel(
    seq_len=seq_len, embed_size=8, nhead=2,
    dim_feedforward=1024, dropout=0, device=device
)

# 3. Load the trained weights
model.load_state_dict(torch.load("forecasting_model_weights.pth", weights_only=True))
model.eval() # Set model to evaluation mode (disables dropout, etc.)
print("Model loaded successfully!")

# 4. Prepare some dummy/validation data to start the forecast
# (In a real scenario, you'd load real historical data here)
x = np.linspace(0, 10, 1000)
val_data = np.sin(x) + np.random.normal(0, 0.05, 1000)

# 5. Inference Loop
forecasted_values = []
current_sequence = val_data[-seq_len:] # Start with the last 200 points

for _ in range(FORECAST_STEPS):
    # Format input for the model: (batch_size=1, seq_len=200, features=1)
    model_input = torch.tensor(current_sequence, dtype=torch.float32).reshape(1, seq_len, 1).to(device)
    
    with torch.no_grad(): # Don't track gradients during inference
        prediction = model(model_input)
        
    predicted_value = prediction.item()
    forecasted_values.append(predicted_value)
    
    # Slide the window forward: drop the oldest value, append the new prediction
    current_sequence = np.append(current_sequence[1:], predicted_value)

print("Forecasting complete!")

# Optional: Plot the results
plt.plot(range(FORECAST_STEPS), forecasted_values, 'r--', label="Predicted")
plt.plot(range(FORECAST_STEPS), np.sin(np.linspace(10, 12, FORECAST_STEPS)), 'g-', label="Actual Ideal")
plt.legend()
plt.show()