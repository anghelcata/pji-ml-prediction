# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 11:36:10 2024

@author: AC
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load the dataset
data_dir = Path("D:/ownCloud/Articol Foisor/csv2")
df = pd.read_csv(data_dir / "foisor_resampled.csv")

# Split the data into features and target variable
X = df.drop(columns=['septic'])  # Features (all columns except 'septic')
y = df['septic']                 # Target variable ('septic')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Gaussian Naive Bayes classifier with the best parameter
best_gnb_model = GaussianNB(var_smoothing=1e-09)

# Train the model
best_gnb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_gnb_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()  # Unpacking True Negatives, False Positives, False Negatives, True Positives

# Calculate specificity
specificity = tn / (tn + fp)

# Print the performance metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
print(f"Specificity: {specificity:.4f}")

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plotting the ROC curve
y_pred_prob = best_gnb_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label="ROC curve (area = {:.4f})".format(roc_auc))
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Save ROC curve data to a CSV file
roc_data = pd.DataFrame({
    'False_Positive_Rate': fpr,
    'True_Positive_Rate': tpr,
    'Thresholds': thresholds
})

# Save the DataFrame to a CSV file for later use
data_res = Path("D:/ownCloud/Articol Foisor/results2")
roc_data.to_csv(data_res / 'GNB_roc_data.csv', index=False)

# Display the confusion matrix in a table format
print("\nConfusion Matrix:")
print('  TP    -  FN')
print(f'  {tp}    -  {fn}')
print(f'  {fp}     -  {tn}')
print('  FP    -  TN')
