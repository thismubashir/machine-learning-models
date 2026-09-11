
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
import os

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "avg-household-size.csv"))

print(df.head())
print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.describe())
print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# Data Cleaning
df = df.drop_duplicates()
df = df.dropna()

print("Shape after cleaning:", df.shape)

# Correlation
correlation = df.corr(numeric_only=True)
print(correlation)

plt.figure(figsize=(7, 5))
plt.imshow(correlation, cmap="viridis", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45)
plt.yticks(range(len(correlation.columns)), correlation.columns)
plt.title("Correlation Matrix")
plt.show()

# X and y
X = df[["countyfips"]]
y = df["avghouseholdsize"]

print("X:")
print(X.head())

print("y:")
print(y.head())

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("Training:", X_train.shape)
print("Testing:", X_test.shape)

# Linear Regression
model = LinearRegression()

# Fit
model.fit(X_train, y_train)

# Coefficients
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])

print(
    f"Equation: y = {model.intercept_:.4f} + "
    f"({model.coef_[0]:.4f})x"
)

# Predict
y_pred = model.predict(X_test)

print("Predictions:")
print(y_pred[:10])

# Actual vs Predicted
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print(results.head(10))

# Residual Error
residuals = y_test.values - y_pred

print("Residuals:")
print(residuals[:10])

# Metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

# Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()
