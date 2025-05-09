# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 17:35:20 2024

@author: AC
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load the CSV file into a DataFrame
data_dir = Path("D:/ownCloud/Articol Foisor/csv2")
data = pd.read_csv(data_dir / 'foisor_final_numerical.csv', low_memory=False)

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Select features that have a correlation above a certain threshold with 'septic'
correlation_with_target = correlation_matrix['septic'].abs()
relevant_features = correlation_with_target[correlation_with_target > 0.015].index.tolist()

# Ensure 'septic' appears as the last column
if 'septic' in relevant_features:
    relevant_features.remove('septic')
    relevant_features.append('septic')

# Filter the correlation matrix to include only relevant features
filtered_corr_matrix = correlation_matrix.loc[relevant_features, relevant_features]

# Create a heatmap
plt.figure(figsize=(16, 14))  # Adjust the size of the plot
sns.heatmap(filtered_corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, linecolor='black')

plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
plt.yticks(rotation=0)  # Ensure y-axis labels are not rotated
plt.title("Correlation Heatmap of Relevant Features", fontsize=18)
plt.show()
