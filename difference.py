# Given MSE values for GMDH and regression models
mse_gmdh = 3.561976679748538e-06
mse_regression_w = 1.510297792372733

# Calculate the percentage improvement
improvement_percentage = (1 - (mse_gmdh / mse_regression_w)) * 100
improvement_percentage
print(f'The GMDH model has a {improvement_percentage:.2f}% improvement over the regression model.')