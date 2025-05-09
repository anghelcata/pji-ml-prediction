# pji-ml-prediction
Predicting Periprosthetic Joint Infection with Machine Learning
This repository contains the source code and methodology used in the research article:

"Predicting Periprosthetic Joint Infection: Evaluating Supervised Machine Learning Models for Clinical Application"
Manuscript ID: JOTr-D-25-00307

🔍 Overview
The project applies multiple supervised machine learning algorithms to predict the risk of periprosthetic joint infection (PJI) using structured electronic health record data from 27,854 patients treated at “Foisor” Orthopaedic Clinical Hospital.

Algorithms evaluated:

Random Forest (RF)
XGBoost
Artificial Neural Network (ANN)
Logistic Regression
AdaBoost
Gaussian Naive Bayes
k-Nearest Neighbors (kNN)
Stochastic Gradient Descent (SGD)
🗂 Repository Structure
pji-ml-prediction/
├── data/                 # Data folder (not included for privacy reasons)
├── results/              # Saved figures, SHAP plots, ROC data
├── src/
│   ├── preprocess/       # Preprocessing scripts (cleaning, imputation, SMOTE-ENN)
│   ├── models/           # Classifier scripts for each ML model
│   ├── eda/              # Exploratory data analysis scripts
│   └── eval/             # Combined evaluation scripts (e.g., ROC overlay)
├── requirements.txt      # Python dependencies
├── README.md             # Project overview and instructions
└── LICENSE               # Open-source license
