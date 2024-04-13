import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from joblib import dump, load
from gmdhpy import gmdh
# Ensure the save_model function is correctly imported
from ModelSerialization import save_model, load_model
from sklearn.linear_model import LinearRegression
import joblib


# After training your model (assuming `model` is your trained LinearRegression object)
joblib.dump(model, 'linear_regression_model.joblib')

# Assuming the model has been trained and saved as 'linear_regression_model.joblib'
model = joblib.load('linear_regression_model.joblib')
scaler = joblib.load('scaler.joblib')  # Assuming the scaler has been saved too

# Set up the page
st.title('Steel Alloy Strength Prediction')

# User input
st.subheader('Enter the chemical composition of the steel alloy:')
c = st.number_input('Carbon (C) content:')
mn = st.number_input('Manganese (Mn) content:')
si = st.number_input('Silicon (Si) content:')
# Repeat for all features...

# Button to make predictions
if st.button('Calculate Mechanical Properties'):
    # Create a DataFrame from the input features
    features = pd.DataFrame([[c, mn, si, ...]], columns=['c', 'mn', 'si', ...])
    # Scale the features using the saved scaler
    features_scaled = scaler.transform(features)

    # Make predictions
    predictions = model.predict(features_scaled)
    # Display predictions
    st.subheader('Predicted Mechanical Properties:')
    st.write('Yield Strength:', predictions[0][0])
    st.write('Tensile Strength:', predictions[0][1])
    st.write('Elongation:', predictions[0][2])
