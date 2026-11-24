# 💳 Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using classification algorithms and class imbalance handling techniques.

---

## 📌 Problem Statement

Credit card fraud is a major financial threat. This project builds a binary classifier to distinguish between **fraudulent (Class = 1)** and **legitimate (Class = 0)** transactions from anonymized transaction data.

---

## 📂 Dataset

The dataset used is the well-known [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.

| Feature | Description |
|--------|-------------|
| `Time` | Seconds elapsed between this and the first transaction |
| `Amount` | Transaction amount |
| `V1–V28` | Anonymized PCA-transformed features |
| `Class` | Target variable — `1` = Fraud, `0` = Legitimate |

> ⚠️ The dataset is **highly imbalanced** — fraudulent transactions make up only ~0.17% of all records.

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Libraries:**
  - `pandas`, `numpy` — Data manipulation
  - `matplotlib`, `seaborn` — Visualization
  - `scikit-learn` — Preprocessing, model training, evaluation
  - `imbalanced-learn` — SMOTE for oversampling
  - `xgboost` — Gradient boosting classifier

---

## 🔄 Project Workflow

```
1. Load & Explore Data
        ↓
2. Exploratory Data Analysis (EDA)
        ↓
3. Data Preprocessing (Feature Scaling)
        ↓
4. Train-Test Split (Stratified)
        ↓
5. Baseline Model — Logistic Regression
        ↓
6. Handle Class Imbalance with SMOTE
        ↓
7. Retrain Model on Balanced Data
        ↓
8. Evaluate (Confusion Matrix, Classification Report)
```

---

## 📊 Exploratory Data Analysis

- Class distribution plot to visualize the severe imbalance
- Correlation heatmap across all features to identify relationships

---

## ⚙️ Preprocessing

- `Amount` and `Time` are **standardized** using `StandardScaler`
- Original `Time` and `Amount` columns are dropped after scaling
- Scaled features (`scaled_amount`, `scaled_time`) are placed at the front of the dataframe

---

## 🤖 Models Used

| Model | Notes |
|-------|-------|
| Logistic Regression (Baseline) | Trained on raw imbalanced data |
| Logistic Regression + SMOTE | Retrained on SMOTE-balanced data with `class_weight='balanced'` |

**Planned / Future Models:**
- Random Forest Classifier
- XGBoost Classifier
- ROC-AUC Score comparison across models

---

## 📈 Evaluation Metrics

Given the class imbalance, accuracy alone is misleading. The following metrics are used:

- **Confusion Matrix**
- **Precision, Recall, F1-Score** (Classification Report)
- **ROC-AUC Score** *(planned)*

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

### 3. Download the dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root directory.

### 4. Run the notebook
```bash
jupyter notebook Credit_Card_Fraud_Detection.ipynb
```

---

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── Credit_Card_Fraud_Detection.ipynb   # Main notebook
├── creditcard.csv                       # Dataset (download separately)
└── README.md                            # Project documentation
```

---

## 🔮 Future Work

- Implement and compare **Random Forest** and **XGBoost** classifiers
- Add **ROC-AUC curve** visualization for model comparison
- Tune hyperparameters using `GridSearchCV`
- Deploy as a simple web app using Flask or Streamlit

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
