
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

from xgboost import XGBRegressor
# 2. Load Dataset

df = pd.read_csv("/content/Food_Delivery_Times.csv")

print(df.head())
print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())
target = "Delivery_Time_min"
# 5. Handle Missing Values

df = df.dropna()

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# 6. Delivery Time Distribution

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Delivery_Time_min"],
    kde=True
)

plt.title("Delivery Time Distribution")
plt.xlabel("Delivery Time (minutes)")
plt.ylabel("Number of Deliveries")

plt.show()


plt.figure(figsize=(8, 5))

sns.boxplot(
    x="Traffic_Level",
    y="Delivery_Time_min",
    data=df
)

plt.title("Traffic Level vs Delivery Time")

plt.show()


# 8. Distance vs Delivery Time

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x="Distance_km",
    y="Delivery_Time_min",
    data=df
)

plt.title("Distance vs Delivery Time")

plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (minutes)")

plt.show()

X = df.drop(
    target,
    axis=1
)

y = df[target]

X = pd.get_dummies(X)


# 11. Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
# 12. Original XGBoost Model

model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)


# 13. Train Original Model

model.fit(
    X_train,
    y_train
)


# 14. Original Train Prediction

y_train_pred = model.predict(
    X_train
)


# 15. Original Test Prediction

y_test_pred = model.predict(
    X_test
)


# 16. Original Train Metrics

train_mae = mean_absolute_error(
    y_train,
    y_train_pred
)

train_r2 = r2_score(
    y_train,
    y_train_pred
)


# 17. Original Test Metrics

test_mae = mean_absolute_error(
    y_test,
    y_test_pred
)

test_r2 = r2_score(
    y_test,
    y_test_pred
)


# 18. Original Model Results

print("Train MAE:", train_mae)
print("Test MAE:", test_mae)

print("Train R2:", train_r2)
print("Test R2:", test_r2)


# 19. Feature Importance

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(
    ascending=False
)

print("\nTop 10 Features:")
print(importance.head(10))


# 20. Feature Importance Plot

importance.head(10).plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Top 10 Important Features")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.show()
# 21. Hyperparameter Search Space

param_grid = {
    "n_estimators": [100, 200, 300, 400, 500],

    "learning_rate": [0.01, 0.03, 0.05, 0.1],

    "max_depth": [2, 3, 4, 5, 6],

    "min_child_weight": [1, 3, 5, 7],

    "subsample": [0.7, 0.8, 0.9, 1.0],

    "colsample_bytree": [0.7, 0.8, 0.9, 1.0]
}


# 22. Randomized Search

random_search = RandomizedSearchCV(
    estimator=XGBRegressor(
        random_state=42
    ),

    param_distributions=param_grid,

    n_iter=30,

    cv=3,

    scoring="neg_mean_absolute_error",

    random_state=42,

    n_jobs=-1
)


random_search.fit(
    X_train,
    y_train
)
print(
    random_search.best_params_
)
best_model = random_search.best_estimator_
# 26. Tuned Train Prediction

y_train_pred_tuned = best_model.predict(
    X_train
)


# 27. Tuned Test Prediction

y_test_pred_tuned = best_model.predict(
    X_test
)


# 28. Tuned Train Metrics

train_mae_tuned = mean_absolute_error(
    y_train,
    y_train_pred_tuned
)

train_r2_tuned = r2_score(
    y_train,
    y_train_pred_tuned
)


# 29. Tuned Test Metrics

test_mae_tuned = mean_absolute_error(
    y_test,
    y_test_pred_tuned
)

test_r2_tuned = r2_score(
    y_test,
    y_test_pred_tuned
)


# 30. Tuned Model Results


print("Train MAE:", train_mae_tuned)

print("Test MAE:", test_mae_tuned)

print("Train R2:", train_r2_tuned)

print("Test R2:", test_r2_tuned)


# 31. Generalization Gap

mae_gap = test_mae_tuned - train_mae_tuned

r2_gap = train_r2_tuned - test_r2_tuned


print("MAE Gap:", mae_gap)

print("R2 Gap:", r2_gap)


# 32. Generalization Interpretation

if r2_gap < 0.05:
    print("Model is generalizing well.")

elif r2_gap < 0.10:
    print("Model has a moderate generalization gap.")

else:
    print("Model may be overfitting.")
# 33. Actual vs Predicted

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    y_test_pred_tuned
)

plt.xlabel("Actual Delivery Time")
plt.ylabel("Predicted Delivery Time")

plt.title("Tuned Model: Actual vs Predicted")

plt.show()


# 34. Tuned Feature Importance

tuned_importance = pd.Series(
    best_model.feature_importances_,
    index=X.columns
)

tuned_importance = tuned_importance.sort_values(
    ascending=False
)

print("\nTop 10 Features - Tuned Model:")

print(
    tuned_importance.head(10)
)


# 35. Tuned Feature Importance Plot

tuned_importance.head(10).plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title(
    "Top 10 Important Features - Tuned Model"
)

plt.xlabel("Features")
plt.ylabel("Importance")

plt.show()


# 36. Final Comparison

print("\nOriginal Model:")

print("Train MAE:", train_mae)
print("Test MAE:", test_mae)
print("Train R2:", train_r2)
print("Test R2:", test_r2)

print("\nTuned Model:")

print("Train MAE:", train_mae_tuned)
print("Test MAE:", test_mae_tuned)
print("Train R2:", train_r2_tuned)
print("Test R2:", test_r2_tuned)

print("\nGeneralization:")

print("MAE Gap:", mae_gap)
print("R2 Gap:", r2_gap)
