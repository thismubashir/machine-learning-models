import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
df = pd.read_csv("/content/heart_disease_uci.csv")

print(df.head())

print("\nShape:", df.shape)

print("\nMissing Values:\n", df.isnull().sum())
df.drop_duplicates(inplace=True)

df["target"] = (df["num"] > 0).astype(int)

df.drop(columns=["num", "id", "dataset"], inplace=True)

print("\nTarget Distribution:")

print(df["target"].value_counts())
plt.figure(figsize=(6, 4))

sns.countplot(x="target", data=df)

plt.title("Heart Disease Distribution")

plt.xlabel("Target")

plt.ylabel("Count")

plt.show()


categorical_columns = df.select_dtypes(include="object").columns

df = pd.get_dummies(

    df,

    columns=categorical_columns,

    drop_first=True

)

X = df.drop("target", axis=1)

y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)


X_train = X_train.fillna(X_train.median())

X_test = X_test.fillna(X_train.median())