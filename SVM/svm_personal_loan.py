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

print(df["Personal Loan"].value_counts(normalize=True) * 100)
sns.countplot(x="Personal Loan", data=df)

plt.title("Personal Loan Distribution")

plt.xlabel("Personal Loan")

plt.ylabel("Number of Customers")

plt.show()


plt.figure(figsize=(12, 8))

sns.heatmap(

    df.corr(numeric_only=True),

    annot=True,

    cmap="coolwarm"

)

plt.title("Correlation Heatmap")

plt.show()


plt.figure(figsize=(7, 5))

sns.boxplot(

    x="Personal Loan",

    y="Income",

    data=df

)

plt.title("Income vs Personal Loan")

plt.show()


plt.figure(figsize=(7, 5))

sns.boxplot(

    x="Personal Loan",

    y="Age",

    data=df

)

plt.title("Age vs Personal Loan")

plt.show()


plt.figure(figsize=(7, 5))

sns.boxplot(

    x="Personal Loan",

    y="CCAvg",

    data=df

)

plt.title("Credit Card Average Spending vs Personal Loan")

plt.show()


sns.countplot(

    x="Education",

    hue="Personal Loan",

    data=df

)

plt.title("Education vs Personal Loan")

plt.show()


sns.countplot(

    x="Family",

    hue="Personal Loan",

    data=df

)

plt.title("Family Size vs Personal Loan")

plt.show()
df = df.drop(columns=["ID", "ZIP Code"])

X = df.drop(columns=["Personal Loan"])

y = df["Personal Loan"]


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)