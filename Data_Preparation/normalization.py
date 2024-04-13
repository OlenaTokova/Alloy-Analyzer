import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset from the specified path
data_path = '/Users/Elena/Documents/GitHub/steel_strength/metals_data.xlsx'  # Update this path to where your file is located
data = pd.read_excel(data_path)

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler to the dataset and transform the data
data_normalized = scaler.fit_transform(data)

# Convert the normalized array back to a DataFrame for easier handling
data_normalized_df = pd.DataFrame(data_normalized, columns=data.columns)

# Display the first few rows of the normalized data to verify
print(data_normalized_df.head())
