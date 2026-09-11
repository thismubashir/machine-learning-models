
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

