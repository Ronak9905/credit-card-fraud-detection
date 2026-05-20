# Credit Card Fraud Detection

## Project Overview

This project focuses on building and evaluating multiple machine learning models to detect fraudulent credit card transactions. Credit card fraud detection is a highly challenging classification problem due to the **extreme class imbalance**, where fraudulent transactions represent only a very small percentage of all transactions.

The project explores data preprocessing, feature engineering, threshold optimization, imbalance handling, and multiple machine learning algorithms to maximize fraud detection performance while maintaining a balance between precision and recall.

The primary goal is to reduce **false negatives (missed fraud cases)** while maintaining strong precision to minimize false fraud alerts.

---

## Dataset

The dataset used is `creditcard.csv`, containing credit card transactions made by European cardholders in September 2013.

### Dataset Information

- **Total Transactions:** 284,807
- **Total Features:** 31
- **Target Variable:** `Class`
  - `0` → Legitimate Transaction
  - `1` → Fraudulent Transaction

### Feature Description

- `V1` to `V28` → PCA-transformed anonymized features
- `Time` → Time elapsed between transactions
- `Amount` → Transaction amount
- `Class` → Fraud label

### Dataset Characteristics

- No missing values found
- Highly imbalanced dataset
- Fraudulent transactions represent only a very small portion of total transactions

---

## Exploratory Data Analysis (EDA)

Several exploratory data analysis techniques were performed to understand transaction patterns and identify important predictive features.

### Key Insights

#### 1. Time Distribution
- Transaction patterns over time were analyzed.
- A new feature called `hour` was later extracted from the `Time` column.

#### 2. Amount Distribution
- The `Amount` feature showed a highly skewed distribution.
- Log transformation using `np.log1p()` was applied to reduce skewness and improve model performance.

#### 3. Outlier Analysis
Outliers in normal transactions were explored and filtered during EDA to create a cleaned dataset (`df_cleaned`).

Dataset size reduced from:

```plaintext
(284807, 31) → (252945, 31)
```

However, the final models were trained on the **original dataset after duplicate removal**, while outlier filtering was explored mainly for EDA and analysis purposes to avoid losing potentially important fraud patterns.

#### 4. Correlation Analysis

### Strong Negative Correlation with Fraud
- `V17`
- `V14`
- `V12`
- `V10`
- `V3`
- `V16`
- `V7`
- `V1`
- `V9`
- `V5`

### Positive Correlation with Fraud
- `V11`
- `V4`
- `V2`

These features demonstrated strong predictive power for fraud detection.

---

## Data Cleaning & Preprocessing

The following preprocessing steps were applied before training:

### 1. Duplicate Removal

Duplicate transactions were identified and removed to improve dataset quality.

### 2. Train-Test Split

The dataset was split using an **80:20 ratio** with:

```python
stratify=y
```

This preserved fraud class distribution in both training and testing datasets.

### 3. Feature Engineering

Two additional features were created:

#### `hour`
Extracted from the `Time` feature.

#### `is_zero_amount`
Binary feature indicating whether transaction amount is zero:

```plaintext
1 → Amount = 0  
0 → Amount > 0
```

### 4. Time Column Removal

After extracting useful information, the original `Time` column was removed.

### 5. Log Transformation

The `Amount` feature was transformed using:

```python
np.log1p()
```

to reduce skewness.

### 6. Feature Scaling

Numerical features were standardized using:

```python
StandardScaler
```

to improve model performance and maintain consistent feature ranges.

---

## Models Used

Multiple machine learning models were trained and evaluated for fraud detection.

### 1. Logistic Regression

#### Configuration

- `penalty='l2'`
- `C=1.0`
- `class_weight={0:1,1:4}`
- `max_iter=1000`

#### Threshold Optimization

A custom threshold optimization strategy was implemented to maximize **recall** while ensuring **precision remained above 0.5**.

#### Evaluation Metrics

- Recall
- Precision
- F1-Score
- ROC-AUC

---

### 2. Random Forest Classifier

#### Configuration

- `n_estimators=150`
- `max_depth=10`
- `min_samples_leaf=100`
- `max_samples=0.1`
- `max_features='sqrt'`
- `class_weight='balanced'`

#### Threshold Optimization

Threshold tuning was performed using:

```python
fbeta_score(beta=2)
```

to prioritize fraud detection recall.

#### Evaluation Metrics

- Recall
- Precision
- F1-Score
- ROC-AUC

---

### 3. XGBoost Classifier

#### Configuration

- `n_estimators=500`
- `max_depth=6`
- `learning_rate=0.1`
- `scale_pos_weight=100`

#### Threshold Optimization

Threshold optimized to maintain:

```plaintext
Precision between 0.70–0.75
```

while maximizing recall.

#### Evaluation Metrics

- 5-Fold Stratified Cross-Validation
- Precision
- Recall
- F1-Score
- ROC-AUC

---

### 4. LightGBM Classifier

#### Configuration

- `n_estimators=1000`
- `learning_rate=0.02`
- `num_leaves=63`
- `min_child_samples=20`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `scale_pos_weight=100`

#### Threshold Optimization

Cross-validation threshold tuning prioritized:

- High Recall
- Minimum **75% Precision**

Fallback:

- Best F1-score threshold

#### Evaluation Metrics

- Precision
- Recall
- F1-Score
- ROC-AUC

---

### 5. CatBoost Classifier

#### Configuration

- `iterations=400`
- `learning_rate=0.03`
- `depth=6`
- `auto_class_weights='Balanced'`

#### Cross Validation

5-Fold Cross Validation was performed using **AUC** as the evaluation metric.

#### Threshold Optimization

Threshold optimized using:

```python
F2-score
```

to prioritize recall while balancing precision.

#### Evaluation Metrics

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-Score

---

### 6. Ensemble Model (LightGBM + CatBoost)

#### Methodology

A weighted average ensemble was created:

- **LightGBM Weight:** `0.4`
- **CatBoost Weight:** `0.6`

CatBoost was assigned a higher weight due to its stronger precision performance.

---

## Model Performance Comparison

| Model | Precision | Recall | F1-Score | ROC-AUC |
|--------|------------|---------|-----------|----------|
| Logistic Regression | 0.5862 | 0.8673 | 0.6996 | 0.9660 |
| Random Forest | 0.7304 | 0.8571 | 0.7887 | 0.9723 |
| XGBoost | 0.7049 | 0.8776 | 0.7818 | 0.9720 |
| LightGBM | 0.7521 | 0.8980 | 0.8186 | 0.9769 |
| CatBoost | 0.8317 | 0.8571 | 0.8442 | 0.9782 |
| Ensemble (LGBM + CatBoost) | 0.7589 | 0.8673 | 0.8095 | 0.9782 |

### Best Performing Model

**CatBoost** achieved the strongest overall performance with:

- **Highest Precision:** `0.8317`
- **Highest F1-Score:** `0.8442`
- **Excellent Recall:** `0.8571`
- **High ROC-AUC:** `0.9782`

This makes CatBoost the most balanced and reliable model for fraud detection in this project.

---

## Project Structure

```plaintext
CREDITCARD_FRAUD_DETECTION/
│── Data/
│   └── creditcard.csv
│
│── Models/
│   ├── catboost_model.pkl
│   └── scaler.pkl
│
│── Notebooks/
│   └── credit_card_fraud_detection.ipynb
│
│── README.md
│── requirements.txt
```

---

## Notebook

The complete implementation including preprocessing, feature engineering, model training, threshold optimization, and evaluation is available in:

```plaintext
Notebooks/credit_card_fraud_detection.ipynb
```

The notebook was developed using **Google Colab** and uploaded directly to GitHub in `.ipynb` format.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
```

Navigate to the project directory:

```bash
cd CREDITCARD_FRAUD_DETECTION
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

### Option 1: Run in Google Colab (Recommended)

1. Open **Google Colab**
2. Upload the notebook file:

```plaintext
Notebooks/credit_card_fraud_detection.ipynb
```

3. Upload the dataset:

```plaintext
Data/creditcard.csv
```

4. Install dependencies if needed:

```bash
!pip install lightgbm xgboost catboost
```

5. Run all notebook cells.

---

### Option 2: Run Locally in VS Code

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook file:

```plaintext
Notebooks/credit_card_fraud_detection.ipynb
```

Run all cells using the **VS Code Jupyter extension** or any notebook-compatible environment.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- LightGBM
- CatBoost
- Pickle
- Google Colab

---

## Saved Model & Scaler

The following files were saved for deployment and reuse:

### Saved Model
- `catboost_model.pkl`

### Saved Scaler
- `scaler.pkl`

Both were saved using Python's `pickle` module to avoid retraining.

---

## Future Improvements

- Hyperparameter tuning using Optuna or GridSearchCV
- Real-time fraud detection pipeline
- Model explainability using SHAP
- Streamlit or Flask deployment
- Advanced ensemble learning methods

---

## Conclusion

This project demonstrates a complete end-to-end machine learning workflow for fraud detection, including:

- Data preprocessing
- Feature engineering
- Class imbalance handling
- Threshold optimization
- Model comparison
- Performance evaluation

Special emphasis was placed on minimizing **false negatives (missed fraud cases)** while maintaining strong precision to reduce false fraud alerts.

Among all evaluated models, **CatBoost delivered the best overall performance**, demonstrating strong fraud detection capability with an excellent balance between precision and recall.