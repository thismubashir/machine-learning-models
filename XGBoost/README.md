# Food Delivery Time Prediction --- XGBoost

An end-to-end machine learning project for predicting **food delivery time in minutes** using operational and contextual order data.

The project uses **XGBoost Regression** to learn the relationship between delivery time and factors such as distance, preparation time, traffic, weather, time of day, vehicle type, and courier experience.

---

## Project Overview

Accurate delivery-time prediction can help food-delivery platforms improve:

* Estimated time of arrival (ETA)
* Customer experience
* Courier allocation
* Operational planning
* Delivery performance monitoring
* Traffic and weather-aware scheduling

This project follows a practical machine-learning workflow:

```text
Raw Dataset
    ↓
Data Exploration
    ↓
Missing-Value Analysis
    ↓
Data Cleaning
    ↓
Categorical Encoding
    ↓
Feature / Target Separation
    ↓
Train-Test Split
    ↓
XGBoost Regression
    ↓
Model Evaluation
    ↓
Feature Importance Analysis
    ↓
Prediction Visualization
```

---

## Dataset

The dataset contains **1,000 delivery records and 9 columns**.

### Features

| Feature                | Type        | Description                                           |
| ---------------------- | ----------- | ----------------------------------------------------- |
| Order_ID               | Integer     | Unique order identifier                               |
| Distance_km            | Float       | Delivery distance in kilometers                       |
| Weather                | Categorical | Weather condition during delivery                     |
| Traffic_Level          | Categorical | Traffic level: Low, Medium, or High                   |
| Time_of_Day            | Categorical | Time period such as Morning, Afternoon, Evening, etc. |
| Vehicle_Type           | Categorical | Delivery vehicle type                                 |
| Preparation_Time_min   | Integer     | Food preparation time in minutes                      |
| Courier_Experience_yrs | Float       | Courier experience in years                           |

### Dataset Shape

```text
Rows:    1,000
Columns: 9
```

---

## Data Quality

Initial inspection identified missing values in four columns:

```text
Weather                    30
Traffic_Level              30
Time_of_Day                30
Courier_Experience_yrs    30
```

The missing values were handled during data cleaning.

After cleaning:

```text
Missing values in all columns: 0
```

This ensures that the model receives a complete feature matrix during training.

---

## Machine Learning Problem

This is a **supervised regression problem**.

### Target

```text
Delivery_Time_min
```

The model learns a function approximately represented as:

```text
Delivery Time =
f(
    Distance,
    Weather,
    Traffic,
    Time of Day,
    Vehicle Type,
    Preparation Time,
    Courier Experience
)
```

The goal is to predict a continuous numerical value representing the expected delivery time in minutes.
---

## Model

### XGBoost Regressor

The project uses **XGBoost**, a gradient-boosting algorithm designed for high-performance supervised learning.

XGBoost builds an ensemble of decision trees sequentially. Each new tree attempts to reduce the errors made by the existing ensemble.

Conceptually:

```text
Training Data
     ↓
Tree 1
     ↓
Calculate Errors
     ↓
Tree 2 learns from errors
     ↓
Calculate Errors
     ↓
Tree 3 learns from remaining errors
     ↓
...
     ↓
Final Prediction
```

XGBoost is particularly useful here because delivery time can depend on nonlinear interactions between variables.

For example:

```text
Long Distance + High Traffic
```

may have a much larger effect on delivery time than either factor considered independently.

---

## Categorical Feature Handling

Categorical variables were converted into numerical features before model training.

One-hot encoding creates binary columns such as:

```text
Traffic_Level_High
Traffic_Level_Low

Weather_Clear
Weather_Foggy
Weather_Rainy

Vehicle_Type_Bike
Vehicle_Type_Car

Time_of_Day_Evening
```

This allows the regression model to process categorical information numerically.

---

## Model Performance

The model was evaluated on both training and unseen test data.

| Metric |    Train |     Test |
| ------ | -------: | -------: |
| MAE    | 5.09 min | 6.56 min |
| R²     |    0.876 |    0.791 |

### Mean Absolute Error --- MAE

The test MAE is approximately:

```text
6.56 minutes
```

This means the model's predictions differ from the actual delivery times by about 6.56 minutes on average, using absolute error.

Formula:

```text
MAE = (1/n) Σ |Actual - Predicted|
```

Lower MAE indicates smaller prediction errors.

### R² Score

The test R² is:

```text
0.791
```

An R² of approximately 0.79 indicates that the model explains a substantial portion of the variation in delivery time within the evaluated test dataset.

Formula:

```text
R² = 1 - SS_res / SS_tot
```

---

## Feature Importance

The model's feature-importance analysis shows that delivery distance is the dominant feature.

Top features from the model:

| Rank | Feature                  | Importance |
| ---: | ------------------------ | ---------: |
|    1 | `Distance_km`            |     0.3915 |
|    2 | `Preparation_Time_min`   |     0.0998 |
|    3 | `Traffic_Level_High`     |     0.0810 |
|    4 | `Weather_Clear`          |     0.0798 |
|    5 | `Weather_Foggy`          |     0.0615 |
|    6 | `Traffic_Level_Low`      |     0.0498 |
|    7 | `Vehicle_Type_Bike`      |     0.0487 |
|    8 | `Vehicle_Type_Car`       |     0.0341 |
|    9 | `Time_of_Day_Evening`    |     0.0309 |
|   10 | `Courier_Experience_yrs` |     0.0290 |

### Key Observation

`Distance_km` has an importance of approximately 0.39, substantially higher than the other individual features.

This is consistent with the observed relationship between distance and delivery time: as delivery distance increases, delivery time generally increases.

Feature importance indicates how much a feature contributes to the trained model's predictions. It should not automatically be interpreted as causal evidence.
---

## Model

### XGBoost Regressor

The project uses **XGBoost**, a gradient-boosting algorithm designed for high-performance supervised learning.

XGBoost builds an ensemble of decision trees sequentially. Each new tree attempts to reduce the errors made by the existing ensemble.

Conceptually:

Training Data → Tree 1 → Calculate Errors → Tree 2 learns from errors → Calculate Errors → Tree 3 learns from remaining errors → Final Prediction

XGBoost is useful here because delivery time can depend on nonlinear interactions between variables.

For example, **Long Distance + High Traffic** may have a larger effect on delivery time than either factor considered independently.

---

## Categorical Feature Handling

Categorical variables were converted into numerical features before model training.

One-hot encoding creates binary columns such as `Traffic_Level_High`, `Traffic_Level_Low`, `Weather_Clear`, `Weather_Foggy`, `Weather_Rainy`, `Vehicle_Type_Bike`, `Vehicle_Type_Car`, and `Time_of_Day_Evening`.

This allows the regression model to process categorical information numerically.

---

## Model Performance

The model was evaluated on both training and unseen test data.

| Metric | Train | Test |
|---|---:|---:|
| MAE | 5.09 min | **6.56 min** |
| R² | 0.876 | **0.791** |

### Mean Absolute Error — MAE

The test MAE is approximately **6.56 minutes**.

This means the model's predictions differ from the actual delivery times by about **6.56 minutes on average**, using absolute error.

Formula:

`MAE = (1/n) Σ |Actual - Predicted|`

Lower MAE indicates smaller prediction errors.

### R² Score

The test R² is **0.791**.

An R² of approximately 0.79 indicates that the model explains a substantial portion of the variation in delivery time within the evaluated test dataset.

Formula:

`R² = 1 - SS_res / SS_tot`

---

## Feature Importance

The model's feature-importance analysis shows that **delivery distance** is the dominant feature.

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `Distance_km` | 0.3915 |
| 2 | `Preparation_Time_min` | 0.0998 |
| 3 | `Traffic_Level_High` | 0.0810 |
| 4 | `Weather_Clear` | 0.0798 |
| 5 | `Weather_Foggy` | 0.0615 |
| 6 | `Traffic_Level_Low` | 0.0498 |
| 7 | `Vehicle_Type_Bike` | 0.0487 |
| 8 | `Vehicle_Type_Car` | 0.0341 |
| 9 | `Time_of_Day_Evening` | 0.0309 |
| 10 | `Courier_Experience_yrs` | 0.0290 |

### Key Observation

`Distance_km` has an importance of approximately **0.39**, substantially higher than the other individual features.

This is consistent with the observed relationship between distance and delivery time: as delivery distance increases, delivery time generally increases.

> Feature importance indicates how much a feature contributes to the trained model's predictions. It should not automatically be interpreted as causal evidence.
---

## Exploratory Data Analysis

The project includes several visual analyses to understand the dataset and the relationships between different features.

### 1. Delivery Time Distribution

The histogram shows the distribution of delivery times. The majority of observations are concentrated roughly between **30–80 minutes**, with a smaller number of longer deliveries extending beyond 100 minutes. This helps identify the central range and long-tail behavior of delivery times.

### 2. Traffic Level vs Delivery Time

The box plot compares delivery times across Low, Medium, and High traffic levels. The visualization helps examine differences in the distribution and spread of delivery times under different traffic conditions.

### 3. Distance vs Delivery Time

The scatter plot shows a positive relationship between distance and delivery time. As delivery distance increases, delivery time generally increases as well, although other operational factors create variation around the overall relationship.

### 4. Actual vs Predicted Delivery Time

The actual-vs-predicted plot provides a visual check of model performance. A strong regression model generally produces points close to the conceptual diagonal, where predicted values are close to actual values. The plot shows that the model captures the overall relationship between actual and predicted delivery time, while some observations have larger prediction errors.

---

## Project Structure

The XGBoost project is organized as follows:

machine-learning-models/
│
├── XGBoost/
│   ├── Food_Delivery_Times.csv
│   ├── Food_Delivery_Times.ipynb
│   ├── xgboost_food_delivery.py
│   └── README.md
│
└── ...

---

## Installation

### 1. Clone the Repository

    git clone https://github.com/thismubashir/machine-learning-models.git
    cd machine-learning-models

### 2. Create a Virtual Environment

For macOS / Linux:

    python3 -m venv venv
    source venv/bin/activate

For Windows:

    python -m venv venv
    venv\Scripts\activate

### 3. Install Dependencies

    pip install pandas numpy matplotlib seaborn scikit-learn xgboost

---

## Running the Project

From the project root:

    python XGBoost/xgboost_food_delivery.py

The script performs the complete workflow, including:

1. Loading the dataset
2. Inspecting the data
3. Detecting missing values
4. Cleaning missing values
5. Encoding categorical variables
6. Splitting the dataset
7. Training the XGBoost model
8. Generating predictions
9. Calculating MAE and R²
10. Displaying feature importance
11. Generating evaluation plots

---

## Example Output

The model produced the following results:

    Dataset Shape:
    (1000, 9)

    Missing Values After Cleaning:
    0

    Train MAE:
    5.0943

    Test MAE:
    6.5644

    Train R2:
    0.8760

    Test R2:
    0.7912

Feature importance begins with:

    Distance_km               0.391478
    Preparation_Time_min      0.099814
    Traffic_Level_High        0.080994
    Weather_Clear             0.079760
    Weather_Foggy             0.061489

---

## Evaluation Strategy

The dataset is divided into training and testing subsets.

### Training Set

The training set is used to learn the relationship between input features and delivery time.

### Test Set

The test set is used to evaluate how well the trained model performs on previously unseen observations.

The difference between training and test performance is important for understanding model generalization.

The results were:

    Train MAE ≈ 5.09
    Test MAE  ≈ 6.56

    Train R² ≈ 0.876
    Test R²  ≈ 0.791

The lower test performance compared with training performance indicates a generalization gap, which should be monitored when improving the model.