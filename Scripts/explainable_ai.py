# Explainable AI (XAI) using LIME

# LIME is a tool for making machine learning models more interpretable by generating local explanations.

# Import the necessary libraries
import lime
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
import numpy as np

# Load your dataset (replace 'your_dataset.csv' with the actual dataset file)
data = pd.read_csv('your_dataset.csv')

# Select the features and the target variable
features = data.drop(columns=['target_variable'])
target = data['target_variable']

# Create a LIME explainer
explainer = LimeTabularExplainer(training_data=features.to_numpy(), mode="regression")

# Choose a specific instance for explanation
instance_idx = 0  # Replace with the index of the instance you want to explain
instance = features.iloc[instance_idx].values

# Explain the prediction for the selected instance
explanation = explainer.explain_instance(instance, predict_fn=model.predict)

# Print the explanation
print("Explanation for the selected instance:")
explanation.show_in_notebook()
