# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 13:22:25 2024

@author: AC
"""
# Import necessary libraries
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Load the CSV file into a DataFrame
data_dir = Path("D:/ownCloud/Articol Foisor/csv2")
data = pd.read_csv(data_dir / 'foisor_final_numerical.csv', low_memory=False)



# Assuming 'data' is your DataFrame and 'septic' is the target column
# Separate the features and the target
X = data.drop(columns=['septic'])
y = data['septic']

# Apply SMOTEEN
smoteen = SMOTEENN(random_state=42)
X_resampled, y_resampled = smoteen.fit_resample(X, y)

# Check the distribution of the target variable after resampling
print("Counts of values in the 'septic' column after applying SMOTEEN:")
print(y_resampled.value_counts())

# Combine the resampled features and target into a single DataFrame
resampled_data = pd.DataFrame(X_resampled, columns=X.columns)
resampled_data['septic'] = y_resampled

# Save the resampled dataset to a new CSV file
output_path = data_dir / 'foisor_resampled.csv'
resampled_data.to_csv(output_path, index=False)

print(f"Resampled dataset saved to {output_path}")

rows, columns = resampled_data.shape

# Print the total number of rows and columns

print(f"\nTotal number of rows: {rows}")
print(f"Total number of columns: {columns}")