import pandas as pd

# Load the dataset (replace 'your_dataset.csv' with your dataset file)
df = pd.read_csv('your_dataset.csv')

# Handling missing values (replace 'column_name' with the actual column name)
df['column_name'].fillna(df['column_name'].median(), inplace=True)

# Save the preprocessed data to a new CSV file (optional)
df.to_csv('preprocessed_dataset.csv', index=False)

