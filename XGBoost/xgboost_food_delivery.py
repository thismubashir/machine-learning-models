
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


# 7. Traffic vs Delivery Time

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