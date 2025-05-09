# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 18:54:47 2024

@author: AC
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path

# Load the dataset
data_dir = Path("D:/ownCloud/Articol Foisor/csv2")
df = pd.read_csv(data_dir / "foisor_resampled.csv")

# Split the data into features and target variable
X = df.drop(columns=['septic'])  # Features (all columns except 'septic')
y = df['septic']                 # Target variable ('septic')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features using StandardScaler for better performance of ANN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the ANN model
model = Sequential()

# Input layer
model.add(Dense(units=128, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(BatchNormalization())  # Batch Normalization
model.add(Dropout(0.3))  # Dropout to prevent overfitting

# Hidden layer 1
model.add(Dense(units=64, activation='relu'))
model.add(BatchNormalization())  # Batch Normalization
model.add(Dropout(0.3))  # Dropout to prevent overfitting

# Hidden layer 2
model.add(Dense(units=32, activation='relu'))
model.add(BatchNormalization())  # Batch Normalization
model.add(Dropout(0.3))  # Dropout to prevent overfitting

# Output layer
model.add(Dense(units=1, activation='sigmoid'))  # Sigmoid for binary classification

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mse', #binary_crossentropy
              metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train_scaled, y_train, 
                    epochs=500, 
                    batch_size=64, 
                    validation_data=(X_test_scaled, y_test), 
                    callbacks=[early_stopping], 
                    verbose=0
                    )

# Make predictions on the test set
y_pred_prob = model.predict(X_test_scaled).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

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
#roc_data.to_csv(data_res / 'ANN_roc_data.csv', index=False)

# Display the confusion matrix in a table format
print("\nConfusion Matrix:")
print('  TP    -  FN')
print(f'  {tp}    -  {fn}')
print(f'  {fp}     -  {tn}')
print('  FP    -  TN')
