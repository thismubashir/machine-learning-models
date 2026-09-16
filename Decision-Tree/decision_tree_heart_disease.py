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