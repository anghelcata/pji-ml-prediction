# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 10:48:10 2024

@author: AC
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Load the CSV file into a DataFrame
data_dir = Path("D:/ownCloud/Articol Foisor/csv2")
data = pd.read_csv(data_dir / 'foisor_resampled.csv', low_memory=False)

# Display the first few rows of the dataset
print("First five rows of the dataset:")
print(data.head())

# Get a summary of statistics for numerical columns
print("\nSummary statistics:")
print(data.describe())

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Set the size of the plot to be similar to the one you uploaded
plt.figure(figsize=(15, 12))

# Create a heatmap for the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, linewidths=1, linecolor='black')

# Title of the heatmap
plt.title('Correlation Matrix Heatmap')

# Display the heatmap
plt.show()

# Focus on the correlation of the 'septic' column with other features
plt.figure(figsize=(8, 12))
sns.heatmap(correlation_matrix[['septic']].sort_values(by='septic', ascending=False), annot=True, cmap='coolwarm', center=0, linewidths=1, linecolor='black')
plt.title("Correlation of Features with 'septic'")
plt.show()
