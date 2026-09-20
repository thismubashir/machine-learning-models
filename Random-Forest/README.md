# Random Forest — Blood Cancer Classification

A machine learning classification project that uses a **Random Forest Classifier** to predict blood cancer diagnosis from patient-related features.

## Overview

This project demonstrates an end-to-end machine learning workflow for a binary classification problem using the **Random Forest algorithm**.

The workflow includes:

* Dataset loading and inspection
* Missing-value analysis
* Categorical feature encoding
* Train/test splitting
* Random Forest model training
* Accuracy evaluation
* Out-of-Bag (OOB) score evaluation
* Confusion matrix analysis
* Classification report
* Feature importance analysis
* Visualization of model results

## Project Structure

```text
Random-Forest/
│
├── blood_cancer.csv
├── random_forest_blood_cancer.py
├── random_forest_blood_cancer.ipynb
└── README.md
```

## Dataset

The project uses `blood_cancer.csv`.

The dataset contains patient-related features, including:

* `gender`
* `diagnosis`
* Other numerical patient features

The `diagnosis` column is used as the target variable.

The `gender` feature is converted into numerical values using `LabelEncoder` before training the model.

## Machine Learning Workflow

### 1. Data Loading

The dataset is loaded using Pandas and inspected using:

* First few rows
* Dataset shape
* Missing values
* Diagnosis class distribution

### 2. Data Preprocessing

The categorical `gender` feature is converted into numerical form using `LabelEncoder`.

The dataset is then divided into:

* **X** → Input features
* **y** → Target (`diagnosis`)

### 3. Train-Test Split

The data is divided into:

* **80% training data**
* **20% testing data**

Stratified splitting is used to preserve the target-class distribution.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

## Model

The project uses `RandomForestClassifier` from Scikit-learn.

### Model Configuration

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    max_features="sqrt",
    oob_score=True,
    random_state=42
)
```

### Key Parameters

| Parameter      |  Value | Purpose                           |
| -------------- | -----: | --------------------------------- |
| `n_estimators` |    100 | Number of decision trees          |
| `max_depth`    |      5 | Maximum depth of each tree        |
| `max_features` | `sqrt` | Features considered at each split |
| `oob_score`    |   True | Enables Out-of-Bag evaluation     |
| `random_state` |     42 | Reproducible results              |

## Evaluation

The trained model is evaluated using multiple metrics.

### Accuracy

Measures the overall percentage of correctly classified samples.

### OOB Score

Random Forest can evaluate samples that were not selected for a particular tree during bootstrap sampling. This is known as the **Out-of-Bag score**.

### Confusion Matrix

The confusion matrix shows the number of:

* True Positives
* True Negatives
* False Positives
* False Negatives

### Classification Report

The classification report provides:

* Precision
* Recall
* F1-score
* Support

## Feature Importance

The project also analyzes feature importance using:

```python
rf.feature_importances_
```

The features are sorted according to their importance, and the **top 10 features** are visualized using a bar chart.

This helps identify which input features contributed most to the Random Forest's predictions.

## Visualizations

The project includes visualizations for:

1. Confusion Matrix
2. Top 10 Feature Importances

These visualizations make model performance and feature contribution easier to interpret.

## Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter Notebook

## Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
```

## How to Run

Clone the repository:

```bash
git clone https://github.com/thismubashir/machine-learning-models.git
```

Navigate to the project:

```bash
cd machine-learning-models/Random-Forest
```

Run the Python script:

```bash
python random_forest_blood_cancer.py
```

The notebook can also be opened using Jupyter Notebook or JupyterLab.

## Reproducibility

A fixed `random_state=42` is used for the train/test split and Random Forest model to make the results reproducible across runs.

## Project Objective

The objective of this project is to demonstrate how an ensemble learning algorithm such as **Random Forest** can be applied to a medical classification dataset while following a structured machine learning workflow.

## Future Improvements

Potential improvements to this project include:

* Hyperparameter tuning
* Cross-validation
* Additional preprocessing
* Class imbalance analysis
* ROC-AUC evaluation
* Precision-Recall analysis
* Model comparison with other classification algorithms
* Improved feature engineering
* Model deployment through an API

## Disclaimer

This project is intended for **educational and machine learning practice purposes**. The model should not be used for real-world medical diagnosis or clinical decision-making.




### Project Type

**Machine Learning | Supervised Learning | Classification | Ensemble Learning | Random Forest**
