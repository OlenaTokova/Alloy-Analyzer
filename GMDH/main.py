import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from gmdhpy import gmdh
# Import the serialization functions
from ModelSerialization import save_model, load_model
# Load data from a CSV file
data = pd.read_csv(r'C:\Users\Elena\Documents\GitHub\steel_strength\metals_data.csv')

# Assume the first 13 columns are features and the last three are targets
features = data.iloc[:, :13]
outputs = data.iloc[:, -3:]

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split the scaled features and outputs into training and testing datasets
X_train, X_test, y_train_all, y_test_all = train_test_split(features_scaled, outputs, test_size=0.2, random_state=42)

# Prepare to store models and mse for each target
models = []
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
    print(f'MSE for target {i}:', mse)\
        
    # Initialize and train the GMDH model
    model = gmdh.MultilayerGMDH()
    model.fit(X_train, y_train)

    # Store the model
    models.append(model)

    # Make predictions with the trained model
    predictions = model.predict(X_test)

    # Evaluate the model's performance using Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, predictions)
    mses.append(mse)

    # Print MSE for each target
    print(f'MSE for target {i}:', mse)

# Optionally, print the MSEs
print("MSEs:", mses)

