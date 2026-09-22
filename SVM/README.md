# SVM Personal Loan Prediction

## 📌 Project Overview

This project uses a **Support Vector Machine (SVM)** classifier to predict whether a customer is likely to accept a personal loan offer.

The project focuses on building a complete machine learning workflow, including:

* Data exploration
* Exploratory Data Analysis (EDA)
* Data preprocessing
* Feature scaling
* SVM model training
* Model evaluation
* Confusion matrix
* Classification report

The goal is to understand how SVM performs on a real-world-style customer classification problem and identify areas for further improvement.

---

## 📊 Dataset

The project uses the **Universal Bank dataset**.

### Dataset Information

* **Rows:** 5,000
* **Features:** 14 columns
* **Target:** `Personal Loan`
* **Problem Type:** Binary Classification

### Target Distribution

| Personal Loan      | Percentage |
| ------------------ | ---------: |
| 0 — Did not accept |      90.4% |
| 1 — Accepted       |       9.6% |

The dataset is therefore **imbalanced**, with significantly fewer customers accepting personal loans.

### Important Features

The dataset contains customer information such as:

* Age
* Experience
* Income
* ZIP Code
* Family
* CCAvg
* Education
* Mortgage
* Securities Account
* CD Account
* Online Banking
* Credit Card

---

## ⚙️ Machine Learning Workflow

### 1. Data Loading

The dataset was loaded using Pandas and inspected using:

* `head()`
* `shape`
* `info()`
* `describe()`
* `isnull().sum()`
* `duplicated().sum()`

### 2. Exploratory Data Analysis

Several visualizations were created to understand the dataset and relationships between variables.

The analysis included:

* Personal Loan distribution
* Correlation heatmap
* Income vs Personal Loan
* Age vs Personal Loan
* Credit Card Average Spending vs Personal Loan
* Education vs Personal Loan
* Family Size vs Personal Loan

### 3. Data Preparation

The following columns were removed before model training:

```text
ID
ZIP Code
```

The target variable was:

```text
Personal Loan
```

The remaining variables were used as input features.

### 4. Train-Test Split

The dataset was divided into:

* **80% Training Data**
* **20% Testing Data**

Stratified splitting was used to maintain the class distribution.

### 5. Feature Scaling

Since SVM is sensitive to feature scales, **StandardScaler** was used to standardize the numerical features.

### 6. SVM Model

The model was trained using:

```text
SVC(
    kernel="rbf",
    C=1,
    gamma="scale"
)
```

The **RBF (Radial Basis Function)** kernel was used to capture non-linear relationships in the data.

---

## 📈 Model Performance

The final model achieved the following results on the test set:

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **97.80%** |
| Precision | **97.44%** |
| Recall    | **79.17%** |
| F1 Score  | **87.36%** |

### Confusion Matrix

```text
[[902   2]
 [ 20  76]]
```

|              | Predicted 0 | Predicted 1 |
| ------------ | ----------: | ----------: |
| **Actual 0** |         902 |           2 |
| **Actual 1** |          20 |          76 |

The model correctly classified:

* **902** customers who did not accept the loan
* **76** customers who accepted the loan

It incorrectly classified:

* **2** non-loan customers as loan customers
* **20** loan customers as non-loan customers

---

## 🔍 Classification Report

```text
              precision    recall  f1-score   support

           0       0.98      1.00      0.99       904
           1       0.97      0.79      0.87        96

    accuracy                           0.98      1000
   macro avg       0.98      0.89      0.93      1000
weighted avg       0.98      0.98      0.98      1000
```

---

## 🧠 Key Observations

The model achieved **97.8% accuracy**, but accuracy alone does not tell the complete story.

The dataset is highly imbalanced, with only **9.6%** of customers accepting personal loans.

The model achieved:

* **97.44% precision** for the positive class
* **79.17% recall** for the positive class
* **87.36% F1-score** for the positive class

This means the model performs very well overall, but it still misses some customers who would accept a personal loan.

For this type of business problem, identifying potential customers correctly can be important, so **recall and F1-score should be considered alongside accuracy**.

---

## 🚀 Possible Improvements

The current model provides a strong baseline, but several improvements could be explored.

### Hyperparameter Tuning

Search across different:

* `C`
* `gamma`
* `kernel`

values using cross-validation.

### Handle Class Imbalance

Since only 9.6% of customers belong to the positive class, techniques such as:

* `class_weight="balanced"`
* Stratified Cross-Validation
* Appropriate resampling techniques

could be investigated.

### Feature Engineering

Additional features could be created from existing customer information to potentially improve predictive performance.

### Model Comparison

The SVM model could be compared with:

* Logistic Regression
* K-Nearest Neighbors
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost

### Better Evaluation

Future experiments could include:

* ROC-AUC
* PR-AUC
* Cross-Validation
* Threshold tuning

instead of relying mainly on accuracy.

---

## 📁 Project Structure

```text
SVM/
│
├── svm_personal_loan.py
├── SVM.ipynb
├── UniversalBank.csv
└── README.md
```

### Files

**`svm_personal_loan.py`**
Python implementation of the complete SVM workflow.

**`SVM.ipynb`**
Jupyter Notebook containing data exploration, visualizations, model training and evaluation.

**`UniversalBank.csv`**
Dataset used for the project.

**`README.md`**
Project documentation and results.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

## 🎯 Learning Outcomes

Through this project, I practiced:

* Binary classification
* Exploratory Data Analysis
* Feature preprocessing
* Feature scaling
* Support Vector Machines
* RBF kernel
* Confusion matrices
* Precision, Recall and F1-score
* Classification reports
* Understanding class imbalance
* Interpreting model performance

---

## 📌 Conclusion

This project demonstrates a complete **SVM classification workflow** for predicting personal loan acceptance.

The model achieved **97.8% test accuracy** and an **87.36% F1-score for the positive class**. However, because the dataset is imbalanced, accuracy alone should not be used to judge the model.

The current implementation serves as a strong baseline for further experimentation with hyperparameter tuning, class-imbalance handling, feature engineering and model comparison.

---


