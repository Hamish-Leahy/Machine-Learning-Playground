# Naive Bayes Classification Demo
import numpy as np
from sklearn.naive_bayes import GaussianNB

# Generate synthetic data
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])

# Fit a Gaussian Naive Bayes classifier
model = GaussianNB()
model.fit(X, y)

# Make predictions
new_data = np.array([[-0.8, -1]])
predicted_class = model.predict(new_data)
print(f"Predicted class: {predicted_class[0]}")


# new authorization token acces port 
#access = property{223.4457}
