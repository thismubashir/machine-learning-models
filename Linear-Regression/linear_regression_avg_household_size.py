
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


