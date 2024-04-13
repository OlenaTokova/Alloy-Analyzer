import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from joblib import dump, load
from gmdhpy import gmdh
from GMDH.ModelSerialization import save_model, load_model

# Load data from a CSV file
data = pd.read_csv(r'C:\Users\Elena\Documents\GitHub\steel_strength\metals_data.csv')

# Assume the first 13 columns are features and the last three are targets
features = data.iloc[:, :13]
outputs = data.iloc[:, -3:]

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
dump(scaler, 'scaler.joblib')  # Save the scaler immediately after fitting

# Split the scaled features and outputs into training and testing datasets
X_train, X_test, y_train_all, y_test_all = train_test_split(features_scaled, outputs, test_size=0.2, random_state=42)

mses = []

# Train a separate model for each output column and save each model
for i in range(outputs.shape[1]):
    y_train = y_train_all.iloc[:, i]
    y_test = y_test_all.iloc[:, i]

    model = gmdh.MultilayerGMDH()
    model.fit(X_train, y_train)

    # Save the model with a unique name for each target
    save_model(model, f'gmdh_model_target_{i}.joblib')

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mses.append(mse)
    print(f'MSE for target {i}:', mse)

# Optionally, print the MSEs
print("MSEs:", mses)

# Load the scaler when needed
scaler = load('scaler.joblib')

# Example for loading a model and predicting new data
# This should be done where X_new is defined
# X_new_transformed = scaler.transform(X_new)
# model_0 = load_model('gmdh_model_target_0.joblib')
# predictions = model_0.predict(X_new_transformed)
