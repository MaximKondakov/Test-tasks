# Lending_club
## Overview
This Jupyter Notebook for a binary classification problem related to loan status prediction. The notebook includes data preprocessing, model training using multiple algorithms, and ensemble for prediction.

## Data Preprocessing:
Merge training data and map categorical features to numerical values.
Define a function to categorize states into regions.
Preprocess data: scaling, encoding, and splitting into training and validation sets.

## Model Training:
Utilize multiple models: LightGBM, XGBoost, CatBoost, PyTorch neural network, and TabNet.
Implement Stratified K-Fold cross-validation for robust model training and evaluation.
Define a neural network architecture and train it using PyTorch with early stopping.

## Model Evaluation and Ensemble:
Evaluate models using metrics like ROC AUC, accuracy, and F1-score.
Combine predictions from different models using weighted averaging to form an ensemble model.
Save the best-performing model during training based on validation loss.

