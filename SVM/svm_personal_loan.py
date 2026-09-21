import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    confusion_matrix,

    classification_report

)
df = pd.read_csv("/content/UniversalBank.csv")


print(df.head())

print(df.shape)

print(df.info())

print(df.describe())

print(df.isnull().sum())

print(df.duplicated().sum())


print(df["Personal Loan"].value_counts())

print(df["Personal Loan"].value_counts(normalize=True) * 100)df = pd.read_csv("/content/UniversalBank.csv")


print(df.head())

print(df.shape)

print(df.info())

print(df.describe())

print(df.isnull().sum())

print(df.duplicated().sum())


print(df["Personal Loan"].value_counts())

print(df["Personal Loan"].value_counts(normalize=True) * 100)