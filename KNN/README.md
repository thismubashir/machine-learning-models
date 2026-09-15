# KNN Diabetes Classification

This project uses the **K-Nearest Neighbors (KNN)** algorithm to predict whether a person has diabetes based on medical measurements.

## Dataset

The dataset contains **768 records** and **9 columns**.

### Features

* Pregnancies
* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI
* Pedigree
* Age

### Target

* `Outcome`

  * `0` = No Diabetes
  * `1` = Diabetes

## Data Preprocessing

The dataset initially contains no actual `NaN` values, but some medical features contain `0` values that can represent missing or unavailable measurements.

The following columns were checked for zero values:

* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI

These zero values were replaced with `NaN` and then filled using the **median**.

The data was split into:

* **80% Training Data**
* **20% Testing Data**

Because KNN is a distance-based algorithm, **StandardScaler** was used to scale the features before training.

## Model

The project tests different K values from **1 to 20** to find a suitable value for the KNN classifier.

The best K based on test accuracy was:

```text
Best K: 8
Best Accuracy: 76.62%
```

The final model was trained using:

```python
KNeighborsClassifier(
    n_neighbors=best_k,
    weights="distance"
)
```

## Model Results

### Final Test Performance

| Metric   |     Result |
| -------- | ---------: |
| Accuracy | **74.68%** |
| ROC-AUC  | **79.47%** |

### Confusion Matrix

```text
[[84 16]
 [23 31]]
```

This means:

* True Negatives: **84**
* False Positives: **16**
* False Negatives: **23**
* True Positives: **31**

### Classification Report

| Class                | Precision | Recall | F1-Score |
| -------------------- | --------: | -----: | -------: |
| No Diabetes (0)      |      0.79 |   0.84 |     0.81 |
| Diabetes (1)         |      0.66 |   0.57 |     0.61 |
| **Overall Accuracy** |           |        | **0.75** |

The model performs better at identifying the **No Diabetes** class than the **Diabetes** class. The recall for the Diabetes class is **57%**, meaning some diabetes cases were missed by the model.

## Cross Validation

A **5-Fold Cross Validation** was also performed.

```text
Cross Validation Scores:

[0.7236  0.7642  0.6992  0.7561  0.7869]
```

### Mean CV Accuracy

```text
74.60%
```

The cross-validation results are relatively close to the final test accuracy, which gives a better indication of how the model performs across different training/testing splits.

## Visualizations

The project includes:

* K value vs Accuracy graph
* Confusion Matrix
* Model evaluation results

The K value graph was used to compare model performance for different values of K.

## Files

```text
KNN/
├── knn_diabetes.py
├── diabetes.csv
├── KNN.ipynb
└── README.md
```

### `knn_diabetes.py`

Contains the complete Python implementation of the KNN classification model.

### `diabetes.csv`

The dataset used for training and testing the model.

### `KNN.ipynb`

Jupyter Notebook containing the complete analysis, model training, evaluation and visualizations.

## What I Practiced

Through this project, I practiced:

* Loading and exploring a dataset
* Identifying missing values
* Handling zero values as missing data
* Median imputation
* Train-test splitting
* Feature scaling
* KNN classification
* Selecting the K value
* Confusion matrix
* Classification report
* ROC-AUC
* Cross-validation
* Model performance analysis

## Conclusion

The final KNN model achieved an accuracy of **74.68%** and a ROC-AUC score of **79.47%** on the test set.

The model achieved a **74.60% mean accuracy** through 5-fold cross-validation, showing reasonably consistent performance across different validation splits.

However, the model's recall for the Diabetes class was **57%**, so there is still room for improvement. Different preprocessing techniques, feature selection, or other classification algorithms could potentially improve the results.

> **Note:** This project is for machine learning practice and educational purposes only. It should not be used for medical diagnosis.
