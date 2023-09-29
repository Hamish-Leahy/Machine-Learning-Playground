import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset (replace 'your_dataset.csv' with your dataset file)
df = pd.read_csv('your_dataset.csv')

# Encoding categorical variables (replace 'categorical_column' with the actual column name)
label_encoder = LabelEncoder()
df['categorical_column'] = label_encoder.fit_transform(df['categorical_column'])

# Save the preprocessed data to a new CSV file (optional)
df.to_csv('preprocessed_dataset.csv', index=False)

