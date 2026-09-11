# Linear Regression — Average Household Size

This project implements a Linear Regression model to predict the average household size using the county FIPS code.

## Dataset

The dataset contains information about:

* State FIPS
* County FIPS
* Average Household Size
* Geography

## Features

**Input Feature:**

* `countyfips`

**Target:**

* `avghouseholdsize`

## Workflow

1. Load the dataset
2. Inspect the data
3. Clean the data
4. Analyze correlations
5. Split data into training and testing sets
6. Train the Linear Regression model
7. Make predictions
8. Calculate evaluation metrics
9. Analyze residual errors
10. Visualize model predictions and residuals

## Model Evaluation

The model was evaluated using MAE, MSE, RMSE, and R² Score.

| Metric   |  Result |
| -------- | ------: |
| MAE      |  0.1894 |
| MSE      |  0.0735 |
| RMSE     |  0.2711 |
| R² Score | -0.0011 |

### Result

The model achieved an R² score of approximately **-0.0011**, indicating that the selected feature (`countyfips`) does not provide useful predictive power for estimating average household size with this Linear Regression model.

## Visualizations

The project includes:

* Correlation Matrix
* Actual vs Predicted Plot
* Residual Plot
* Residual Distribution

## Files

* `linear_regression_avg_household_size.py` — Python implementation
* `LINEAR_REGRESSION_AVG_HOUSEHOLD_SIZE.ipynb` — Jupyter Notebook
* `avg-household-size.csv` — Dataset

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook
