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
rf = RandomForestClassifier(

    n_estimators=100,

    max_depth=5,

    max_features="sqrt",

    oob_score=True,

    random_state=42

)



rf.fit(X_train, y_train)



y_pred = rf.predict(X_test)



accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)



print("OOB Score:", rf.oob_score_)
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")

print(cm)

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.colorbar()

plt.show()



print("\nClassification Report:")

print(classification_report(y_test, y_pred))




importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": rf.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print("\nFeature Importance:")

print(importance)



top10 = importance.head(10)

plt.figure(figsize=(10, 6))

plt.barh(

    top10["Feature"],

    top10["Importance"]

)

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.title("Top 10 Important Features")

plt.gca().invert_yaxis()

plt.show()