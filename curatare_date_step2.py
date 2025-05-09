# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 11:02:48 2024

@author: AC
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Load the CSV file into a DataFrame
data_dir = Path("D:/ownCloud/Articol Foisor/csv2")
data = pd.read_csv(data_dir / 'foisor_en_cleaned.csv')

# Display the first few rows of the dataset
print("First five rows of the dataset:")
print(data.head())

# Get a summary of statistics for numerical columns
print("\nSummary statistics:")
print(data.describe())

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())


# Step 2: Handle outliers in numeric columns (e.g., bmi)
# Replace extreme outliers with NaN in the bmi column
data['bmi'] = pd.to_numeric(data['bmi'], errors='coerce')
data.loc[(data['bmi'] < 10) | (data['bmi'] > 100), 'bmi'] = np.nan

# Step 3: Impute missing values in numeric columns with the mean
data['bmi'] = data['bmi'].fillna(data['bmi'].mean())
data['weight'] = data['weight'].fillna(data['weight'].mean())
data['height'] = data['height'].fillna(data['height'].mean())

# Optionally, impute other columns if needed
# For example: Fill missing 'albumin' with the mean (if relevant)
# data['albumin'] = data['albumin'].fillna(data['albumin'].mean())

# Step 4: Save the cleaned dataset
output_file_path_cleaned = data_dir / 'foisor_en_cleaned_v2.csv'
data.to_csv(output_file_path_cleaned, index=False)

print(f"Cleaned DataFrame has been successfully saved to {output_file_path_cleaned}")
# Verificarea valorilor lipsă după imputare
print("\nMissing values after handling:")
print(data.isnull().sum())