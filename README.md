<h1 align="center">💳 Credit Card Fraud Detection</h1>

<p align="center">
Machine Learning Project for Detecting Fraudulent Credit Card Transactions using Advanced Classification Models
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![CatBoost](https://img.shields.io/badge/CatBoost-Best%20Model-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

---

## 📌 Project Overview

This project focuses on detecting fraudulent credit card transactions using multiple machine learning models on a highly imbalanced dataset.

Credit card fraud detection is a critical financial security problem where minimizing missed fraud cases while avoiding excessive false alarms is essential.

The primary goal of this project was to maximize fraud detection **recall** while maintaining strong **precision** to reduce false fraud alerts.

---

## 🚀 Key Highlights

✅ Built an end-to-end machine learning pipeline for fraud detection  
✅ Applied feature engineering and preprocessing techniques  
✅ Handled severe class imbalance using class weights and threshold optimization  
✅ Compared multiple machine learning models for performance evaluation  
✅ Achieved strong fraud detection performance using CatBoost Classifier  

---

## 📊 Dataset

The project uses the **Kaggle Credit Card Fraud Detection dataset** containing European card transactions from September 2013.

🔗 Dataset Link:  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Dataset Summary

- **284,807 transactions**
- **31 features**
- **Highly imbalanced target variable**
- Fraud transactions represent a very small percentage of all transactions

### Features

- `V1` to `V28` → PCA-transformed anonymized features  
- `Time` → Time elapsed between transactions  
- `Amount` → Transaction amount  
- `Class` → Fraud label (`0 = Legitimate`, `1 = Fraud`)  

---

## ⚙️ Preprocessing & Feature Engineering

The following preprocessing steps were applied:

- Duplicate removal  
- Stratified train-test split  
- Log transformation on `Amount`  
- Feature scaling using `StandardScaler`  
- Feature engineering:
  - `hour`
  - `is_zero_amount`

Class imbalance handling and threshold optimization were also performed to improve fraud detection performance.

---

## 🔄 Project Workflow

```plaintext
Dataset
   ↓
Data Cleaning & Preprocessing
   ↓
Feature Engineering
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Threshold Optimization
   ↓
Model Evaluation & Comparison
```

---

## 🤖 Models Used

- Logistic Regression  
- Random Forest  
- XGBoost  
- LightGBM  
- CatBoost  
- Ensemble Model (LightGBM + CatBoost)

---

## 📈 Model Performance

Due to severe class imbalance, model performance was evaluated primarily using:

- Precision
- Recall
- F1-Score
- ROC-AUC

| Model | Precision | Recall | F1-Score | ROC-AUC |
|--------|------------|---------|-----------|----------|
| Logistic Regression | 0.5862 | 0.8673 | 0.6996 | 0.9660 |
| Random Forest | 0.7304 | 0.8571 | 0.7887 | 0.9723 |
| XGBoost | 0.7049 | 0.8776 | 0.7818 | 0.9720 |
| LightGBM | 0.7521 | 0.8980 | 0.8186 | 0.9769 |
| CatBoost | **0.8317** | 0.8571 | **0.8442** | **0.9782** |
| Ensemble | 0.7589 | 0.8673 | 0.8095 | 0.9782 |

---

## 🏆 Best Performing Model

### CatBoost Classifier

CatBoost achieved the strongest overall balance between precision and recall:

- **Precision:** `0.8317`
- **Recall:** `0.8571`
- **F1-Score:** `0.8442`
- **ROC-AUC:** `0.9782`

This made CatBoost the best-performing model for this fraud detection task.

---

## 📁 Project Structure

```plaintext
CREDITCARD_FRAUD_DETECTION/
│── Data/
│── Models/
│── Notebooks/
│   └── credit_card_fraud_detection.ipynb
│── .gitignore
│── LICENSE
│── README.md
│── requirements.txt
```

> Dataset and trained model files are excluded from the repository due to file size limitations.

---

## 🛠️ Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-Learn  
- XGBoost  
- LightGBM  
- CatBoost  
- Matplotlib  
- Seaborn  

---

## ▶️ How to Run

Clone the repository:

```bash
git clone https://github.com/Ronak9905/credit-card-fraud-detection.git
cd CREDITCARD_FRAUD_DETECTION
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Open and run the notebook:

```plaintext
Notebooks/credit_card_fraud_detection.ipynb
```

You can run the project using:

- Jupyter Notebook
- Google Colab
- VS Code

---

## 🚀 Future Improvements

- Hyperparameter tuning using Optuna/GridSearchCV  
- Model explainability using SHAP  
- Real-time fraud detection pipeline  
- Streamlit/Flask deployment  
- Advanced ensemble techniques  

---

## 📌 Conclusion

This project demonstrates an end-to-end machine learning workflow for fraud detection, including:

- Data preprocessing
- Feature engineering
- Imbalance handling
- Threshold optimization
- Model comparison

Among all evaluated models, **CatBoost delivered the strongest overall performance**, making it the best-performing model for this fraud detection task.

---