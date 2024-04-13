# Steel Alloy Properties Prediction using GMDH
The GMDH approach is especially suited for complex and nonlinear data, making it ideal for the intricate relationships in materials science. The project is an advanced predictive tool designed to forecast the mechanical properties of steel castings based on their chemical compositions. Utilizing state-of-the-art machine learning algorithms, the system inputs raw material characteristics—specifically, the alloying elements and calculates key performance metrics of the final product. This project utilizes the Group Method of Data Handling (GMDH) to model and predict material properties. The GMDH approach is especially suited for complex and nonlinear data, making it ideal for the intricate relationships in materials science.
# Features
Predictive Modeling: Uses GMDH neural networks to predict material properties based on chemical components. Feature Selection: Automatically identifies and selects significant features, improving model efficiency. Model Serialization: Models are trained, saved, and loaded using Joblib for easy deployment and testing.
# Requirements
Python 3.8 or later pandas NumPy scikit-learn joblib gmdhpy
pip install pandas numpy scikit-learn joblib gmdhpy
# Installation
Clone this repository to your local machine:
git clone https://github.com/yourusername/Alloy-Analyzer cd
To load and use a saved model for prediction: python prediction.py
prediction.py loads a model and applies it to new data for generating predictions. Ensure you update this script to match your specific prediction needs.
# Data Format
Your dataset should be in a CSV format The last three columns should represent the target properties.
# Example Data
# Contributing
Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.
