# Credit Card Fraud Detection with Machine Learning

## About the Project

This project aims to detect fraudulent credit card transactions using Machine Learning techniques.

Fraud detection is a real-world problem faced by financial institutions, where the main challenge is identifying rare fraudulent transactions within a highly imbalanced dataset.

---

## Objective

Build and evaluate machine learning models to classify transactions as fraudulent or non-fraudulent, focusing on performance metrics suitable for imbalanced data.

---

## Dataset


The dataset contains anonymized credit card transactions, including:

- Time
- Amount
- V1 to V28 (PCA-transformed features)
- Class (0 = normal, 1 = fraud)

The dataset is highly imbalanced:
- ~99.8% normal transactions
- ~0.2% fraudulent transactions

Dataset available on Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## Exploratory Data Analysis

- Distribution of transaction classes
- Analysis of transaction amounts
- Identification of class imbalance

---

## Feature Engineering

- Log transformation applied to the "Amount" feature to improve distribution
- Data scaling using StandardScaler

---

## Models Used

- Logistic Regression (baseline)
- Random Forest
- XGBoost

---

## Model Evaluation

Due to class imbalance, the following metrics were prioritized:

- Precision
- Recall
- F1-score
- ROC-AUC

### Key Technique

Threshold adjustment was applied to improve fraud detection (recall), showing better performance than default classification.

---

## Results

- Ensemble models (Random Forest, XGBoost) outperformed Logistic Regression
- Threshold tuning significantly improved fraud detection
- High accuracy alone proved to be misleading in this context

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib

---

## Key Learnings

- Accuracy is not reliable for imbalanced datasets
- Recall is critical in fraud detection problems
- Threshold tuning can improve model performance
- Real-world data requires business-oriented evaluation

---

## Author

Diego Pablo  
[LinkedIn](https://www.linkedin.com/in/diego-pablo/)

[Portfolio](https://diego-pablo.vercel.app/)
