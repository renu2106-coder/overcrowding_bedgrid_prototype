# Logistic Regression Pipeline (Clean Version)

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score


# 🔹 Load dataset
df = pd.read_excel("C:\\Users\\91918\\OneDrive\\Desktop\\Overcrowding_Alert_System.xlsx")
print("Columns:\n", df.columns)

# 🔹 Clean data
df = df.dropna()

# 🔹 Feature engineering
df['load_per_staff'] = df['Current_Patient_Count'] / (df['Staff_Availability'] + 1)
df['emergency_pressure'] = df['Incoming_Emergency_Cases'] / (df['Current_Patient_Count'] + 1)

# 🔹 Features and target
features = [
    'Current_Patient_Count',
    'Bed_Occupancy_Rate',
    'Staff_Availability',
    'Average_Waiting_Time',
    'Incoming_Emergency_Cases',
    'load_per_staff',
    'emergency_pressure'
]

X = df[features]
y = df['Overcrowded']

# 🔹 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔹 Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000))
])

# 🔹 Hyperparameter tuning
param_grid = {
    'clf__C': [0.01, 0.1, 1, 10, 50],
    'clf__solver': ['lbfgs', 'liblinear'],
    'clf__class_weight': [None, 'balanced']
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# 🔹 Best model
best_model = grid.best_estimator_

# 🔹 Predictions
y_pred = best_model.predict(X_test)

# 🔹 Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 🔹 SAVE MODEL (IMPORTANT - LAST STEP)
joblib.dump(best_model, "model.pkl")

print("\n✅ Model saved as model.pkl")