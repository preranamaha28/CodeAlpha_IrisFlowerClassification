

"""Import Libraries"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

"""Load the Dataset"""

df = pd.read_csv("Iris (1).csv")

print("Dataset Loaded Successfully")
print(df.head())

"""Explore the Dataset"""

print("\nShape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSpecies Count:")
print(df['Species'].value_counts())

"""Visualize the Data"""

sns.pairplot(df, hue='Species')
plt.show()

"""Select Features and Target"""

X = df[['SepalLengthCm',
        'SepalWidthCm',
        'PetalLengthCm',
        'PetalWidthCm']]

y = df['Species']

"""Split Data into Training and Testing Sets"""

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", len(X_train))
print("Testing Data:", len(X_test))

"""Train the Machine Learning Model"""

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

print("Model Trained Successfully")

"""Make Predictions"""

y_pred = model.predict(X_test)

print("Predictions:")
print(y_pred)

"""Calculate Accuracy"""

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

"""Generate Classification Report"""

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

"""Generate Confusion Matrix"""

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

"""Visualize Confusion Matrix"""

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

"""Test with a New Flower"""

sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

print("Predicted Species:", prediction[0])