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
model = DecisionTreeClassifier(

    max_depth=6,

    min_samples_split=5,

    min_samples_leaf=4,

    class_weight="balanced",

    random_state=42

)

model.fit(X_train, y_train)


prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("\nDecision Tree Accuracy:", accuracy)
cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

print("\nCross Validation Scores:")
print(cv_scores)
print("Average CV Accuracy:", cv_scores.mean())

print("\nClassification Report:")
print(classification_report(y_test, prediction, target_names=["No Disease", "Disease"]))
cm = confusion_matrix(y_test, prediction)

plt.figure(figsize=(6, 5))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


importance = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})

importance = importance.sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(importance.head(10))


plt.figure(figsize=(8, 6))

sns.barplot(data=importance.head(10), x="Importance", y="Feature")

plt.title("Top 10 Important Features")

plt.show()


plt.figure(figsize=(20, 10))

plot_tree(model, feature_names=X.columns, class_names=["No Disease", "Disease"], filled=True, max_depth=3)

plt.title("Decision Tree")

plt.show()