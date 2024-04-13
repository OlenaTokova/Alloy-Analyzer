import pandas as pd
from joblib import load
import numpy as np

# Load your model
model_path = 'gmdh_model_target_2.joblib'
model = load(model_path)

# Assuming 'X_new' is your new data on which you want predictions
# You'll need to prepare 'X_new' similar to how your training data was prepared
X_new = np.array([...])  # Replace [...] with your actual data

# Scenario 1: Export predictions
predictions = model.predict(X_new)
predictions_df = pd.DataFrame(predictions, columns=['Predictions'])
predictions_df.to_csv('model_predictions.csv', index=False)

# Scenario 2: Export coefficients (if applicable)
# This will depend on whether your model type has coefficients
# For example, if it's a linear model:
if hasattr(model, 'coef_'):
    coefficients = model.coef_
    coeffs_df = pd.DataFrame(coefficients, columns=['Coefficients'])
    coeffs_df.to_csv('model_coefficients.csv', index=False)
