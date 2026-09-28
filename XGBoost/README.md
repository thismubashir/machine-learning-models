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
