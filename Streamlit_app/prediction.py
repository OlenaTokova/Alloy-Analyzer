import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data[iris.target != 2]  # Binary classification for logistic regression
y = iris.target[iris.target != 2]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit a logistic regression model
reg = LogisticRegression()
reg.fit(X_train_scaled, y_train)

# Save the model to a file
joblib.dump(reg, 'regression_model.joblib')

# Save model coefficients and intercept to CSV
coefficients = reg.coef_[0]
intercept = reg.intercept_[0]
coef_df = pd.DataFrame(coefficients, columns=['Coefficients'], index=iris.feature_names[:len(coefficients)])
coef_df.loc['intercept'] = intercept
coef_df.to_csv('model_coefficients.csv')

# Make predictions
predictions = reg.predict(X_test_scaled)

# Save predictions to CSV
predictions_df = pd.DataFrame(predictions, columns=['Predictions'])
predictions_df.to_csv('model_predictions.csv', index=False)

# Optional: Load the model
reg_loaded = joblib.load('regression_model.joblib')

# Use the loaded model to make predictions (demonstration)
loaded_predictions = reg_loaded.predict(X_test_scaled)
print("Predictions from loaded model:", loaded_predictions)
