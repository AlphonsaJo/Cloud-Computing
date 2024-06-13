import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

# Load the dataset
data = pd.read_csv("vmCloud_data.csv")

# Drop rows with missing values for simplicity
data = data.dropna()

# Define features and target variable
X = data[['cpu_usage', 'memory_usage', 'network_traffic', 'power_consumption', 'num_executed_instructions', 'execution_time']]
y = data['energy_efficiency']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
lr_regressor = LinearRegression()

# Perform cross-validation
cv_scores = cross_val_score(lr_regressor, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

# Convert negative MSE to positive values for interpretation
cv_mse_scores = -cv_scores
cv_rmse_scores = np.sqrt(cv_mse_scores)

print("\nCross-Validation Results")
print("Mean MSE: ", np.mean(cv_mse_scores))
print("Standard Deviation of MSE: ", np.std(cv_mse_scores))
print("Mean RMSE: ", np.mean(cv_rmse_scores))
print("Standard Deviation of RMSE: ", np.std(cv_rmse_scores))

# Train the model on the entire training set
lr_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred_lr = lr_regressor.predict(X_test)

# Evaluate the model
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print("\nLinear Regression")
print("Mean Squared Error:", mse_lr)
print("R-squared:", r2_lr)

# Save the Linear Regression model for later use
joblib.dump(lr_regressor, 'lr_model.pkl')


