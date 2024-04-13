import pandas as pd

# Load the dataset
# data = pd.read_csv('metals_data.csv', decimal=',')  # Adjust path as necessary
#data = pd.read_excel('"C:/Users/Elena/Documents/GitHub/steel_strength/metals_data.csv.xlsx"', decimal=',')

data = pd.read_excel(r'C:/Users/Elena/Documents/GitHub/steel_strength/metals_data.xlsx')


# Check for missing values
print(data.isnull().sum())

# Fill missing values if necessary
data.fillna(data.mean(), inplace=True)

# Optionally normalize your data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.drop(['y1', 'y2', 'y3'], axis=1))


# Calculate correlations
correlations = data.corr()
print(correlations)


from sklearn.linear_model import LinearRegression

# Define your independent variables for each model
X = data.drop(['y1', 'y2', 'y3'], axis=1)
y1 = data['y1']
y2 = data['y2']

# Initialize the models
model_y1 = LinearRegression()
model_y2 = LinearRegression()

# Fit the models
model_y1.fit(X, y1)
model_y2.fit(X, y2)

# Print the coefficients
print("Coefficients for y1 model:", model_y1.coef_)
print("Coefficients for y2 model:", model_y2.coef_)


from sklearn.metrics import mean_squared_error

# Making predictions
y1_predictions = model_y1.predict(X)
y2_predictions = model_y2.predict(X)

# Evaluating the models
y1_mse = mean_squared_error(y1, y1_predictions)
y2_mse = mean_squared_error(y2, y2_predictions)

print(f"Mean Squared Error for y1: {y1_mse}")
print(f"Mean Squared Error for y2: {y2_mse}")


import matplotlib.pyplot as plt

# Example: Plot actual vs. predicted values for y1
plt.figure(figsize=(10, 6))
plt.scatter(y1, y1_predictions)
plt.xlabel('Actual y1 values')
plt.ylabel('Predicted y1 values')
plt.title('Actual vs. Predicted y1 Values')
plt.show()

