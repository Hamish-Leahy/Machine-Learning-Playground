from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# this demo uses the GridSearchCV files

# Define your machine learning model (change this to your specific model)
model = RandomForestClassifier()

# Define hyperparameter grid for tuning
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create GridSearchCV object
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5, verbose=2, n_jobs=-1)

# Load your training data (X_train and y_train)

# Perform hyperparameter tuning
grid_search.fit(X_train, y_train)

# Display the best hyperparameters and corresponding model score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Hyperparameters: ", best_params)
print("Best Score: ", best_score)

# Access the best model using grid_search.best_estimator_
best_model = grid_search.best_estimator_

# Train the best model on the entire training set (optional)
best_model.fit(X_train, y_train)

# Make predictions with the best model
# y_pred = best_model.predict(X_test)
