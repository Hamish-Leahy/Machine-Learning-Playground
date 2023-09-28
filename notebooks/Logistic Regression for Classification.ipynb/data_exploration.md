# Data Exploration

In this section, we will perform a comprehensive exploration of the "Iris" dataset to gain insights into its characteristics, understand the distribution of data, and visualize the relationships between features. Data exploration is a vital step in understanding the dataset and preparing it for the subsequent application of logistic regression for classification.

## Overview of the Iris Dataset

Let's start by loading the "Iris" dataset and getting an initial sense of its structure:

```python
# Load the Iris dataset from scikit-learn
from sklearn.datasets import load_iris
iris = load_iris()

# Create a DataFrame for easy data manipulation (replace 'data' and 'target' accordingly)
import pandas as pd
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Display the first few rows of the dataset
df.head()

