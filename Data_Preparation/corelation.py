import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns  # For better visualization of correlation matrix

# Load the dataset
data_path = '/Users/Elena/Documents/GitHub/steel_strength/metals_data.xlsx'  # Update this path
data = pd.read_excel(data_path)

#data = pd.read_cvs()

# Normalize the dataset
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)
data_normalized_df = pd.DataFrame(data_normalized, columns=data.columns)

# Analyze the correlation matrix
correlation_matrix = data_normalized_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Variables')
plt.show()

# Split the dataset into features and target variable
# Assuming the target variable is the last column
X = data_normalized_df.iloc[:, :-1]  # Features
y = data_normalized_df.iloc[:, -1]   # Target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Plotting actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Actual vs. Predicted Values')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.show()
