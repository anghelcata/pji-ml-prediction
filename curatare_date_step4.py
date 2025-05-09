# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 10:50:22 2024

@author: AC
"""

import pandas as pd
from pathlib import Path

# Define path for the dataset
data_dir = Path("D:/ownCloud/Articol Foisor/csv2")

# Load the dataset
data = pd.read_csv(data_dir / 'foisor_en_cleaned_final.csv')

# Function to extract the most recent numeric value and eliminate the date
def extract_latest_value(cell):
    if pd.isna(cell):
        return None
    try:
        latest_measurement = cell.split(',')[-1]  # Take the last entry
        return float(latest_measurement.split(':')[-1].strip())  # Extract and convert to float
    except:
        return None

# Apply the extraction function to each relevant column
columns_to_impute = ['fibrinogen', 'c_reactive_protein', 'esr', 'wbc']
for column in columns_to_impute:
    data[column] = data[column].apply(extract_latest_value)

# Enhanced Imputation logic
for column in columns_to_impute:
    # Calculate the mean for septic == 1 and septic == 0
    mean_septic_1 = data[data['septic'] == 1][column].mean()
    mean_septic_0 = data[data['septic'] == 0][column].mean()

    # Fallback: if mean_septic_1 or mean_septic_0 is NaN, fallback to the overall mean
    overall_mean = data[column].mean()
    if pd.isna(mean_septic_1):
        mean_septic_1 = overall_mean
    if pd.isna(mean_septic_0):
        mean_septic_0 = overall_mean

    # Impute missing values where septic == 1 with the mean of septic == 1
    data.loc[(data['septic'] == 1) & (data[column].isna()), column] = mean_septic_1

    # Impute missing values where septic == 0 with the mean of septic == 0
    data.loc[(data['septic'] == 0) & (data[column].isna()), column] = mean_septic_0

    # Fallback for any remaining missing values (rare edge case)
    data[column].fillna(overall_mean, inplace=True)

# Display the first few rows of the updated columns to check the results
print(data[['fibrinogen', 'c_reactive_protein', 'esr', 'wbc']].head())

# Check missing values after imputation
print("\nMissing values after handling:")
print(data[columns_to_impute].isnull().sum())

# Save the cleaned and imputed dataset
output_file_path_cleaned = data_dir / 'foisor_en_cleaned_imputed_final.csv'
data.to_csv(output_file_path_cleaned, index=False)

print(f"Cleaned and imputed dataset saved to {output_file_path_cleaned}")
