# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 10:41:35 2024

@author: AC
"""

import pandas as pd
from pathlib import Path

# Set pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Load the CSV file into a DataFrame
data_dir = Path("D:/ownCloud/Articol Foisor/csv2")
data = pd.read_csv(data_dir / 'foisor_en.csv', low_memory=False)
# Drop unnecessary columns
data = data.drop(columns=['doctor','insurance_status', 'number', 
                          'admission_diagnosis_text', 
                          'discharge_diagnosis_text', 
                          'admission_diagnosis_code', 'anticoagulant_history'])


# Fill missing values without using inplace=True
data['ct_history'] = data['ct_history'].fillna('No exam')
data['mri_history'] = data['mri_history'].fillna('No exam')
data['fracture_history'] = data['fracture_history'].fillna('No history')
data['implant_history'] = data['implant_history'].fillna('No implants')
data['puncture_history'] = data['puncture_history'].fillna('No puncture')
data['antibiotic_recommendations_history'] = data['antibiotic_recommendations_history'].fillna('No history')
data['xray_history'] = data['xray_history'].fillna('No Record')
data['cardiovascular_history'] = data['cardiovascular_history'].fillna('No History')
data['antibiotic_history'] = data['antibiotic_history'].fillna('No Record')
data['transfusion_history'] = data['transfusion_history'].fillna('No Record')
data['other_pathologies_history'] = data['other_pathologies_history'].fillna('No Pathology')
data['suture_history'] = data['suture_history'].fillna('No Suture')
data['exercise_history'] = data['exercise_history'].fillna('No Exercise')


# Specifică calea pentru a salva fișierul CSV curățat
output_file_path_cleaned = data_dir / 'foisor_en_cleaned.csv'

# Salvează DataFrame-ul curățat într-un nou fișier CSV
data.to_csv(output_file_path_cleaned, index=False)

print(f"Cleaned DataFrame has been successfully saved to {output_file_path_cleaned}")

# Verificarea valorilor lipsă după imputare
print("\nMissing values after handling:")
print(data.isnull().sum())

# Vizualizarea primelor rânduri din dataset-ul curățat
print("\nHead of the cleaned dataset:")
print(data.head())

