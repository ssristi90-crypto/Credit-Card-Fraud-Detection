"""
fraud_detection_no_tf.py
------------------------
End-to-end Credit Card Fraud Detection using Kaggle dataset.
(No TensorFlow version)

Includes:
 - EDA
 - Preprocessing (SMOTE)
 - Model training: Logistic Regression, Random Forest, Isolation Forest
 - Evaluation using Precision, Recall, F1-score
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from imblearn.over_sampling import SMOTE

# -------------------------
# 1. Load dataset
# -------------------------
df = pd.read_csv("creditcard.csv")

print("Data shape:", df.shape)
print(df.head())

# -------------------------
# 2. EDA (basic plots)
# -------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df)
plt.title("Class Distribution (0 = Non-Fraud, 1 = Fraud)")
plt.show()

fraud_ratio = df['Class'].mean() * 100
print(f"Fraudulent transactions: {fraud_ratio:.4f}%")

plt.figure(figsize=(8, 4))
sns.histplot(df['Amount'], bins=50)
plt.title("Transaction Amount Distribution")
plt.show()

# -------------------------
# 3. Preprocessing
# -------------------------
X = df.drop('Class', axis=1)
y = df['Class']

# Scale Amount and Time columns
scaler = StandardScaler()
X['scaled_amount'] = scaler.fit_transform(X['Amount'].values.reshape(-1, 1))
X['scaled_time'] = scaler.fit_transform(X['Time'].values.reshape(-1, 1))
X = X.drop(['Amount', 'Time'], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle class imbalance with SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print("After SMOTE:", np.bincount(y_train_res))

# ----------------------
