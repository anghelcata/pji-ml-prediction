# RF_final.py
import pandas as pd
import time
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)

# Set paths
data_dir = Path("D:/ownCloud/Articol_Foisor_PJI/csv2")
data_res = Path("D:/ownCloud/Articol_Foisor_PJI/results2")

# Load data
df = pd.read_csv(data_dir / "foisor_resampled.csv")
df = df.drop(columns=['admission_date'], errors='ignore')
X = df.drop(columns=["septic"])
y = df["septic"]

# Train-test split
X_train, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_test = pd.DataFrame(X_test_raw, columns=X.columns)  # ensure column names preserved

# Train Random Forest
model = RandomForestClassifier(
    bootstrap=False,
    max_depth=30,
    min_samples_leaf=1,
    min_samples_split=5,
    n_estimators=200,
    random_state=42
)

start_train = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start_train
print(f"Training time: {training_time:.4f} seconds")

# Predict and timing
start_pred = time.time()
y_pred = model.predict(X_test)
prediction_time = time.time() - start_pred
print(f"Prediction time (total): {prediction_time:.6f} seconds")
print(f"Prediction time per sample: {prediction_time / len(X_test):.8f} seconds")

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn + fp)

# Print metrics
print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
print(f"Specificity: {specificity:.4f}")

# Save confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(data_res / "confusion_matrix_rf.png")
plt.show()

# ROC curve
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(data_res / "roc_curve_rf.png")
plt.show()

# Save ROC data
roc_data = pd.DataFrame({
    "False_Positive_Rate": fpr,
    "True_Positive_Rate": tpr,
    "Thresholds": thresholds
})
roc_data.to_csv(data_res / "RF_roc_data.csv", index=False)

# SHAP analysis — fixed for classification with Explanation object
print("Generating SHAP summary bar plot...")
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Extract class 1 SHAP values
shap_class1 = shap_values.values[..., 1]  # (samples, features)
mean_shap = np.abs(shap_class1).mean(axis=0)
feature_names = X_train.columns.tolist()

if len(mean_shap) != len(feature_names):
    raise ValueError("Mismatch in SHAP and feature dimensions")

shap_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean_SHAP": mean_shap
})
shap_df_sorted = shap_df.sort_values(by="Mean_SHAP", ascending=False).head(10)

plt.figure(figsize=(10, 8))
sns.barplot(data=shap_df_sorted, x="Mean_SHAP", y="Feature", palette="Blues_d")
plt.xlabel("Mean |SHAP value|")
plt.title("Top 10 Most Influential Features (Random Forest)")
plt.tight_layout()
plt.savefig(data_res / "shap_summary_rf_FINAL.png", dpi=150)
plt.show()

# Confusion matrix summary
print("\nConfusion Matrix:")
print("  TP    -  FN")
print(f"  {tp}    -  {fn}")
print(f"  {fp}     -  {tn}")
print("  FP    -  TN")
