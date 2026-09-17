# Heart Disease Prediction using Decision Tree

A machine learning classification project that predicts whether a patient is likely to have **heart disease** using a **Decision Tree Classifier**.

The project covers the complete machine learning workflow, including data preprocessing, categorical encoding, missing-value handling, model training, cross-validation, classification evaluation, feature importance analysis, and decision-tree visualization.

---

## Project Overview

### Objective

The objective of this project is to build a Decision Tree classification model that predicts whether a patient has heart disease based on clinical and demographic features.

The original dataset contains a `num` variable representing the presence/severity of heart disease. For this project, it is converted into a binary classification target:

| Target | Meaning    |
| ------ | ---------- |
| `0`    | No Disease |
| `1`    | Disease    |

This makes the problem suitable for binary classification.

---

## Machine Learning Workflow

The project follows a structured machine learning pipeline:

```text
Raw Dataset
     │
     ▼
Data Loading
     │
     ▼
Data Inspection
     │
     ▼
Duplicate Removal
     │
     ▼
Target Creation
     │
     ▼
Remove Unnecessary Columns
     │
     ▼
Categorical Encoding
     │
     ▼
Missing Value Handling
     │
     ▼
Train / Test Split
     │
     ▼
Decision Tree Training
     │
     ▼
Prediction
     │
     ├── Accuracy
     ├── Cross-Validation
     ├── Classification Report
     ├── Confusion Matrix
     └── Feature Importance
```

---

## Dataset

The project uses the **Heart Disease UCI dataset**.

The dataset contains patient-related clinical information and a target indicating the presence of heart disease.

### Data Preparation

The following preprocessing steps are applied:

* Duplicate records are removed.
* The original `num` variable is converted into a binary `target`.
* Unnecessary columns such as `id` and `dataset` are removed.
* Categorical features are converted into numerical features using one-hot encoding.
* Missing numerical values are filled using the median of the training data.
* The dataset is divided into training and testing sets using an 80/20 split.
* Stratification is used to preserve the target-class distribution.

---

## Model

The project uses Scikit-learn's `DecisionTreeClassifier`.

The model is configured with the following parameters:

```python
DecisionTreeClassifier(
    max_depth=6,
    min_samples_split=5,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42
)
```

### Why These Parameters?

| Parameter           |    Value | Purpose                                                      |
| ------------------- | -------: | ------------------------------------------------------------ |
| `max_depth`         |        6 | Limits tree depth and helps control complexity               |
| `min_samples_split` |        5 | Requires a minimum number of samples before splitting a node |
| `min_samples_leaf`  |        4 | Prevents very small leaf nodes                               |
| `class_weight`      | balanced | Helps account for class imbalance                            |
| `random_state`      |       42 | Makes results reproducible                                   |

The depth and minimum-sample constraints are used to reduce the risk of creating an unnecessarily complex tree.

---

## Model Evaluation

The model is evaluated using multiple metrics rather than relying only on accuracy.

### Accuracy

Measures the overall percentage of correctly classified observations.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Cross-Validation

Five-fold cross-validation is used to evaluate model consistency across different subsets of the data.

```python
cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)
```

The individual cross-validation scores and their average are reported by the notebook.

### Classification Report

The classification report provides:

* Precision
* Recall
* F1-score
* Support

for both:

* No Disease
* Disease

This provides a more detailed view of classification performance.

### Confusion Matrix

A confusion matrix is generated to visualize:

* True Negatives
* False Positives
* False Negatives
* True Positives

This is particularly useful for understanding the types of classification errors made by the model.

---

## Feature Importance

The Decision Tree provides feature importance values that indicate how much each feature contributes to the tree's decision-making process.

The project extracts feature importance using:

```python
model.feature_importances_
```

The top 10 features are displayed and visualized using a bar chart.

> Feature importance indicates the contribution of features within this trained tree; it should not be interpreted as proof of medical causation.

---

## Decision Tree Visualization

The trained Decision Tree is visualized using Scikit-learn's `plot_tree()`.

The visualization displays:

* Decision rules
* Feature splits
* Class predictions
* Tree structure
* Node information

The visualization is limited to the first three levels for readability.

---

## Visualizations

The project includes several visualizations:

### 1. Target Distribution

Shows the distribution between:

* No Disease
* Disease

### 2. Confusion Matrix

Displays the model's classification results across both classes.

### 3. Feature Importance

Shows the top 10 features according to the trained Decision Tree.

### 4. Decision Tree

Visualizes the learned decision structure of the classifier.

---

## Technologies & Libraries

* **Python**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Jupyter Notebook**

---

## Project Structure

```text
Decision-Tree/
│
├── decision_tree_heart_disease.py
├── heart_disease_uci.csv
├── Heart_Disease_Decision_Tree.ipynb
└── README.md
```

### Files

| File                                | Description                    |
| ----------------------------------- | ------------------------------ |
| `decision_tree_heart_disease.py`    | Complete Python implementation |
| `heart_disease_uci.csv`             | Heart disease dataset          |
| `Heart_Disease_Decision_Tree.ipynb` | Interactive Jupyter Notebook   |
| `README.md`                         | Project documentation          |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/thismubashir/machine-learning-models.git
```

Move into the project:

```bash
cd machine-learning-models/Decision-Tree
```

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

---

## Running the Project

### Python Script

Run:

```bash
python decision_tree_heart_disease.py
```

### Jupyter Notebook

Start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
Heart_Disease_Decision_Tree.ipynb
```

---

## Key Machine Learning Concepts Demonstrated

This project demonstrates practical implementation of:

* Binary classification
* Data cleaning
* Duplicate handling
* Target engineering
* One-hot encoding
* Missing-value imputation
* Stratified train-test splitting
* Decision Tree classification
* Class balancing
* Cross-validation
* Accuracy evaluation
* Precision, Recall and F1-score
* Confusion matrix analysis
* Feature importance
* Model visualization
* Reproducible machine learning

---

## Limitations

This project is intended as a **machine learning learning and portfolio project**, not as a clinical diagnostic system.

Important limitations include:

* Model performance depends on the dataset and preprocessing strategy.
* Feature importance from a Decision Tree does not establish medical causation.
* The model has not been clinically validated.
* Predictions should not be used for real-world medical diagnosis or treatment decisions.
* Additional external validation would be required before considering any clinical application.

---

## Future Improvements

Potential improvements include:

* Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
* Comparing Decision Tree with Random Forest and Gradient Boosting
* Evaluating additional classification metrics
* ROC-AUC analysis
* Precision-Recall analysis
* Feature selection
* Model interpretability using SHAP
* External validation on an independent dataset
* Model serialization for deployment
* Building an API for inference
* Adding automated testing and reproducible experiment tracking

---

## Learning Outcome

This project demonstrates how a raw healthcare dataset can be transformed into a machine learning-ready dataset and used to train, evaluate, and interpret a classification model.

The main focus is not only training the model, but understanding the complete workflow from **data preprocessing to model interpretation**.

---



## Repository

Part of the Machine Learning Models collection:

**machine-learning-models**

The repository contains implementations of multiple machine learning algorithms and projects developed as part of the author's machine learning learning journey.
