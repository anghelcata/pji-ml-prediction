# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 11:16:53 2024

@author: AC
"""

import pandas as pd
from pathlib import Path

# Define path for the dataset
data_dir = Path("D:/ownCloud/Articol Foisor/csv2")

# Load the dataset
data = pd.read_csv(data_dir / 'foisor_en_cleaned_v2.csv')

# Listează coloanele pe care vrei să le elimini
columns_to_drop = [
    'albumin', 'glycated_hemoglobin', 'total_protein', 'urine_culture',
]

# Elimină coloanele din DataFrame
data = data.drop(columns=columns_to_drop)

# Verificarea noii forme a DataFrame-ului
print("Shape of the DataFrame after dropping columns:", data.shape)
print("Remaining columns:")
print(data.columns)

# Salvarea dataset-ului curățat
output_file_path_cleaned = data_dir / 'foisor_en_cleaned_final.csv'
data.to_csv(output_file_path_cleaned, index=False)

print(f"Final cleaned DataFrame has been successfully saved to {output_file_path_cleaned}")
# Verificarea valorilor lipsă după imputare
print("\nMissing values after handling:")
print(data.isnull().sum())

#print(data.head())
rows, columns = data.shape

# Print the total number of rows and columns

print(f"\nTotal number of rows: {rows}")
print(f"Total number of columns: {columns}")