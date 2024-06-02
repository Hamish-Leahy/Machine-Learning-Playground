import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset (replace 'data.csv' with your dataset file)
data = pd.read_csv('data.csv')

# ----- FEATURE ENGINEERING -----

# Optionally add feature engineering steps here (e.g., filling missing values,
# creating new features, encoding categorical variables, etc.)

# ----- SPLITTING AND SCALING -----

# Split the data into features (X) and labels (y)
X = data.drop('target', axis=1)
y = data['target']

# Create train/test split BEFORE scaling
# This prevents data leakage from the test set into the training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (fit on training data only, then transform both)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # Important: Transform only, don't fit again

# ----- SAVING PREPROCESSED DATA -----

# Create a function for consistent saving
def save_data(df, filename, is_label=False):
    header = False if is_label else True
    df.to_csv(filename, index=False, header=header)

# Save the preprocessed data
save_data(pd.DataFrame(X_train), 'X_train.csv')
save_data(pd.DataFrame(X_test), 'X_test.csv')
save_data(pd.Series(y_train), 'y_train.csv', is_label=True)  # Save labels as Series
save_data(pd.Series(y_test), 'y_test.csv', is_label=True) 
