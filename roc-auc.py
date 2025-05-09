# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 13:43:21 2024

@author: AC
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Specify the directory where the CSV files are located
data_dir = Path("D:/ownCloud/Articol Foisor/results2")

# Define the paths to your CSV files (these should be the files you've already saved)
csv_files = [
    'LR_roc_data.csv',
    'RF_roc_data.csv',
    'XGBoost_roc_data.csv',
    'ANN_roc_data.csv',
    'kNN_roc_data.csv',
    'AdaBoost_roc_data.csv',
    'GNB_roc_data.csv',
    'SGD_roc_data.csv'
]

# Define model names to use in the legend
model_names = [
    'Logistic Regression',
    'Random Forest',
    'XGBoost',
    'Artificial Neural Network',
    'k-Nearest Neighbors',
    'AdaBoost',
    'Gaussian Naive Bayes',
    'Stochastic Gradient Descent'
]

# Plot each ROC curve
plt.figure(figsize=(10, 8))

for i, file in enumerate(csv_files):
    # Construct the full path to the CSV file
    file_path = data_dir / file
    
    # Load the data from the CSV file
    data = pd.read_csv(file_path)
    
    # Extract False Positive Rate and True Positive Rate
    fpr = data['False_Positive_Rate']
    tpr = data['True_Positive_Rate']
    
    # Plot the ROC curve for each model
    plt.plot(fpr, tpr, label=model_names[i])

# Plot the diagonal line (random classifier)
plt.plot([0, 1], [0, 1], 'k--')

# Add gridlines
plt.grid(True)

# Add labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

# Show the plot
plt.tight_layout()
plt.show()
