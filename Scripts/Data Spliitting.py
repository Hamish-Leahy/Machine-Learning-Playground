from sklearn.model_selection import train_test_split

# Load your dataset
X, y = load_data()  # Replace with your data loading logic

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the split datasets
save_data(X_train, y_train, 'train_data.csv')  # Replace with your save data function
save_data(X_test, y_test, 'test_data.csv')  # Replace with your save data function
