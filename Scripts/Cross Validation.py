# Cross-Validation in Machine Learning

In machine learning, cross-validation is a crucial technique for assessing a model's performance. It helps in estimating how well a model will generalize to new, unseen data. In this Jupyter Notebook, we will explore various cross-validation strategies.

```python
import numpy as np
from sklearn.model_selection import cross_val_score, KFold

# Sample data and model
X, y = np.array(range(10)), np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
model = YourMachineLearningModel()

# 5-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

# Display cross-validation results
print(f'Cross-Validation Accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})')

