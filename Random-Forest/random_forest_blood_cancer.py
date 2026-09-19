import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (

    accuracy_score,

    confusion_matrix,

    classification_report

)
df = pd.read_csv("/content/blood_cancer.csv")

print(df.head())

print("\nShape:")

print(df.shape)

print("\nMissing Values:")

print(df.isnull().sum())

print("\nDiagnosis:")

print(df["diagnosis"].value_counts())
le = LabelEncoder()

df["gender"] = le.fit_transform(df["gender"])


X = df.drop("diagnosis", axis=1)

y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)