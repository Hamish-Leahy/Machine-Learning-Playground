from sklearn.tree import DecisionTreeClassifier

# Create a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Fit the classifier to your data
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

