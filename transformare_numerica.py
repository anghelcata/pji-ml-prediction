# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 12:39:51 2024

@author: AC
"""

import pandas as pd
from pathlib import Path

# Set pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Load the CSV file into a DataFrame
data_dir = Path("D:/ownCloud/Articol Foisor/csv2")
df = pd.read_csv(data_dir / 'foisor_en_cleaned_imputed_final.csv', low_memory=False)

# 1. Preprocesarea coloanei exercise_history
df['exercise_present'] = df['exercise_history'].apply(lambda x: 0 if 'No Exercise' in x else 1)
df['exercise_count'] = df['exercise_history'].apply(lambda x: 0 if x == 'No Exercise' else len(x.split(',')))

# 2. Preprocesarea coloanei discharge_diagnosis_code
df['diagnosis_present'] = df['discharge_diagnosis_code'].apply(lambda x: 0 if pd.isnull(x) else 1)

# 3. Preprocesarea coloanelor xray_history, ct_history, mri_history
df['xray_present'] = df['xray_history'].apply(lambda x: 0 if 'No exam' in x else 1)
df['ct_present'] = df['ct_history'].apply(lambda x: 0 if 'No exam' in x else 1)
df['mri_present'] = df['mri_history'].apply(lambda x: 0 if 'No exam' in x else 1)

# 4. Preprocesarea coloanei surgery_history
df['surgery_present'] = df['surgery_history'].apply(lambda x: 0 if 'No surgery' in x else 1)
df['surgery_count'] = df['surgery_history'].apply(lambda x: len(x.split(',')) if 'No surgery' not in x else 0)

# 5. Preprocesarea coloanei antibiotic_history
df['antibiotic_present'] = df['antibiotic_history'].apply(lambda x: 0 if 'No history' in x else 1)
df['antibiotic_count'] = df['antibiotic_history'].apply(lambda x: len(x.split(',')) if 'No history' not in x else 0)

# 6. Preprocesarea coloanelor fracture_history, implant_history, puncture_history
df['fracture_present'] = df['fracture_history'].apply(lambda x: 0 if 'No history' in x else 1)
df['implant_present'] = df['implant_history'].apply(lambda x: 0 if 'No implants' in x else 1)
df['puncture_present'] = df['puncture_history'].apply(lambda x: 0 if 'No puncture' in x else 1)

# 7. Preprocesarea coloanelor cardiovascular_history, other_pathologies_history
df['cardiovascular_present'] = df['cardiovascular_history'].apply(lambda x: 0 if 'No History' in x else 1)
df['other_pathologies_present'] = df['other_pathologies_history'].apply(lambda x: 0 if 'No History' in x else 1)

# 8. Columna septic nu necesită modificări, deja este binară

# Salvează datasetul preprocesat
df.to_csv(data_dir / 'preprocessed_dataset.csv', index=False)

print("Preprocesarea a fost realizată cu succes și fișierul a fost salvat!")

print(df.head())

# Identifică coloanele categorice
categorical_columns = df.select_dtypes(include=['object']).columns
print("Coloane categorice:", categorical_columns)

# Transformă manual fiecare coloană categorică
for col in categorical_columns:
    df[col] = pd.factorize(df[col])[0]

# Verifică dacă mai sunt coloane categorice
print(df.select_dtypes(include=['object']).columns)

# Salvează rezultatul într-un nou fișier CSV
df.to_csv(data_dir/ 'foisor_final_numerical.csv', index=False)

print("Toate coloanele categorice au fost transformate și fișierul a fost salvat cu succes!")
print(df.head())


print("\nMissing values in each column:")
print(df.isnull().sum())

#print(data.head())
rows, columns = df.shape

# Print the total number of rows and columns

print(f"\nTotal number of rows: {rows}")
print(f"Total number of columns: {columns}")