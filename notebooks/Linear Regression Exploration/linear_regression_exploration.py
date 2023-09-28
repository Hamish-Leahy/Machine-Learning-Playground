# Linear Regression Exploration
# Author: Your Name
# Date: September 28, 2023

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset (replace 'your_dataset.csv' with your actual dataset file)
data = pd.read_csv('your_dataset.csv')

# Explore the dataset
print("Dataset Summary:")
print(data.head())

# Data preprocessing
X = data['Feature'].values.reshape(-1, 1)
y = data['Target'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Visualize the results
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Linear Regression Model')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression Exploration')
plt.legend()
plt.show()

# Print model performance metrics
print("Model Performance Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Conclusion
print("\nConclusion:")
print("This notebook demonstrates a basic exploration of Linear Regression. It includes loading and preprocessing data, splitting it into training and testing sets, training a Linear Regression model, and evaluating its performance.")

