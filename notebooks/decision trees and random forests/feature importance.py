# Get feature importances
feature_importances = rf_classifier.feature_importances_

# Sort feature importances in descending order
sorted_indices = np.argsort(feature_importances)[::-1]

# Print feature ranking
print("Feature ranking:")
for f in range(10):  # Print the top 10 features
    print(f"{feature_names[sorted_indices[f]]}: {feature_importances[sorted_indices[f]]}")

