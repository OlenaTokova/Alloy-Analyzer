import pandas as pd
from joblib import load

# Load the model
model = load('gmdh_model_target_2.joblib')
print("Model loaded successfully.")

# Assuming you have new data prepared and scaled appropriately in a DataFrame called X_new
# Assume original features were something like this (ensure order and names):
feature_names = ['c', 'mn', 'si', 'cr', 'ni', 'mo', 'v', 'n', 'nb', 'co', 'w', 'al', 'ti']

# Sample new data using the correct feature names:
X_new = pd.DataFrame({
    'c': [0.02], 'mn': [0.05], 'si': [0.05], 'cr': [0.01], 'ni': [19.7], 'mo': [2.95],
    'v': [0.01], 'n': [0], 'nb': [0.01], 'co': [15], 'w': [0], 'al': [0.15], 'ti': [1.55]
})

# Load the scaler and transform the new data
scaler = load('scaler.joblib')
X_new_transformed = scaler.transform(X_new)


# Assuming the scaler is also loaded if the data needs to be scaled
scaler = load('scaler.joblib')  # Make sure the scaler is loaded if your model requires scaled data
X_new_transformed = scaler.transform(X_new)

# Use the model to make predictions
predictions = model.predict(X_new_transformed)
print("Predictions:", predictions)

# Load the model
model = load('gmdh_model_target_2.joblib')

# Make predictions
predictions = model.predict(X_new_transformed)
print("Predictions:", predictions)
