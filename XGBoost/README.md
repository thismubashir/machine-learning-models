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
