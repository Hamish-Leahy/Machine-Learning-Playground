# Advanced Decision Tree Classification Demo

# Import necessary libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import export_text, export_graphviz
import graphviz

# Load the Iris dataset as an example
data = load_iris()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42, max_depth=3)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Visualize the decision tree rules
tree_rules = export_text(clf, feature_names=data.feature_names)
print("\nDecision Tree Rules:")
print(tree_rules)

# Visualize the decision tree graph (requires graphviz)
dot_data = export_graphviz(clf, out_file=None, feature_names=data.feature_names, filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree")  # Save the tree to a file
