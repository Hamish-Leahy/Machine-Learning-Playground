import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset (replace 'your_dataset.csv' with your dataset file)
df = pd.read_csv('your_dataset.csv')

# Select numerical features to be scaled (replace 'numerical_columns' with column names)
numerical_columns = ['feature1', 'feature2', 'feature3']

# Standardize numerical features
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Save the preprocessed data to a new CSV file (optional)
df.to_csv('preprocessed_dataset.csv', index=False)

