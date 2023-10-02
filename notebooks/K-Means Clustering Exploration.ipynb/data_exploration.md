# Data Exploration

In this section, we will perform a comprehensive exploration of the "Iris" dataset to gain insights into its characteristics, understand the distribution of data, and visualize the relationships between features. Data exploration is a fundamental step in preparing the dataset for K-Means clustering and in understanding the inherent structures within the data.
\\\\\\\\\
## Overview of the Iris Dataset

Let's begin by loading the "Iris" dataset and obtaining an initial sense of its structure:

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

