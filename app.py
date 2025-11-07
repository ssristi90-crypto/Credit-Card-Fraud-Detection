# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from imblearn.over_sampling import SMOTE
from collections import Counter

# ----------------------------
# Streamlit Config
# ----------------------------
st.set_page_config(page_title="Credit Card Fraud Detection Dashboard", layout="wide")
st.title("💳 Credit Card Fraud Detection Dashboard")

st.markdown("""
This dashboard demonstrates a **real-world Credit Card Fraud Detection project** using classical machine learning models.
It performs data preprocessing, SMOTE balancing (internally), model training, and evaluation on real imbalanced data.
""")

# -------------------------------------
# Upload Dataset
# -------------------------------------
uploaded_file = st.file_uploader("📂 Upload your 'creditcard.csv' dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # For faster demo on large datasets
    if df.shape[0] > 50000:
        st.warning("⚠️ Large dataset detected — using a 50,000-row random sample for faster training.")
        df = df.sample(50000, random_state=42)

    st.success(f"✅ Dataset Loaded — Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # ----------------------------
    # Exploratory Data Analysis
    # ----------------------------
    st.header("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        class_counts = df['Class'].value_counts()
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        sns.barplot(x=class_counts.index, y=class_counts.values, palette="coolwarm", ax=ax1)
        for i, v in enumerate(class_counts.values):
            ax1.text(i, v + (max(class_counts.values) * 0.02),
                     f"{v:,}\n({v / len(df) * 100:.3f}%)", ha='center', fontweight='bold')
        ax1.set_xticklabels(["Non-Fraud (0)", "Fraud (1)"])
        ax1.set_title("Original Class Distribution")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

        fraud_ratio = df['Class'].mean() * 100
        st.write(f"💡 Fraudulent transactions represent only **{fraud_ratio:.4f}%** of total data — showing strong imbalance.")

    with col2:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.histplot(df['Amount'], bins=40, ax=ax2, color='skyblue', edgecolor='black')
        ax2.set_title("Transaction Amount Distribution")
        ax2.set_xlabel("Transaction Amount")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)

    # ----------------------------
    # Preprocessing
    # ----------------------------
    st.header("⚙️ Data Preprocessing")

    X = df.drop('Class', axis=1)
    y = df['Class']

    scaler = StandardScaler()
    X['scaled_amount'] = scaler.fit_transform(X['Amount'].values.reshape(-1, 1))
    X['scaled_time'] = scaler.fit_transform(X['Time'].values.reshape(-1, 1))
    X = X.drop(['Amount', 'Time'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Safe SMOTE handling
    min_class_count = min(Counter(y_train).values())
    k_neighbors = min(5, max(1, min_class_count - 1))
    sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    st.success(
        f"✅ SMOTE applied internally for model training "
        f"(Balanced {Counter(y_train_res)[0]:,} vs {Counter(y_train_res)[1]:,} samples)."
    )

    # ----------------------------
    # Model Training & Evaluation (cached)
    # ----------------------------
    st.header("🤖 Model Training & Evaluation")

    @st.cache_data
    def train_and_evaluate(X_train_res, y_train_res, X_test, y_test):
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
            "Isolation Forest": IsolationForest(contamination=0.001, random_state=42)
        }

        results = {}
        for name, model in models.items():
            if name == "Isolation Forest":
                preds = model.fit_predict(X_test)
                preds = np.where(preds == -1, 1, 0)
            else:
                model.fit(X_train_res, y_train_res)
                preds = model.predict(X_test)

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, preds, average='binary', zero_division=0
            )
            results[name] = {"Precision": precision, "Recall": recall, "F1-Score": f1}

        return pd.DataFrame(results).T.round(3)

    with st.spinner("⏳ Training models and evaluating performance..."):
        metrics_df = train_and_evaluate(X_train_res, y_train_res, X_test, y_test)

    st.dataframe(metrics_df.style.highlight_max(axis=0, color='#D4EDDA'))

    # ----------------------------
    # Visualization
    # ----------------------------
    st.header("📈 Model Performance Comparison")

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    metrics_df.plot(kind="bar", ax=ax3, rot=0, color=['#2196F3', '#4CAF50', '#FFC107'])
    ax3.set_title("Model Performance Metrics (Evaluated on Real Test Data)")
    ax3.set_ylabel("Score")
    ax3.set_ylim(0, 1)
    for container in ax3.containers:
        ax3.bar_label(container, fmt="%.2f", label_type="edge", fontsize=8)
    st.pyplot(fig3)

    st.markdown("""
    **Interpretation Tips:**
    - Higher Precision → fewer false positives (better for fraud detection).
    - Higher Recall → more frauds detected (sensitive detection).
    - F1-Score balances both.
    """)

else:
    st.info("👆 Upload your `creditcard.csv` dataset to begin analysis.")
