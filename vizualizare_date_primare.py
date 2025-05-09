# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 09:48:15 2024

@author: AC
"""

import pandas as pd
from pathlib import Path

# Load the CSV file into a DataFrame
data_dir = Path("D:/ownCloud/Articol Foisor/csv2")
data = pd.read_csv(data_dir / 'foisor_resampled.csv', low_memory=False)

# Verify the missing values after imputation
print("\nMissing values before handling:")
print(data.isnull().sum())
print("\nData types of remaining columns:")
print(data.dtypes)

#print(data.head())
rows, columns = data.shape

# Print the total number of rows and columns

print(f"\nTotal number of rows: {rows}")
print(f"Total number of columns: {columns}")