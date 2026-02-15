import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef,
    classification_report
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="ML Assignment 2 - Classification App",
    layout="wide"
)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown("""
# Machine Learning Assignment - 2  
### Developed by: **Vaneet Pal Singh**  
**Roll No:** 2025AB05262  
---
""")

st.write("Upload a dataset and evaluate different classification models.")

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV format only)",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # ---------------------------------------------------
    # CLEAN DATA (MUST MATCH TRAINING)
    # ---------------------------------------------------
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", regex=True)]

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.write("Dataset Shape:", df.shape)

    # ---------------------------------------------------
    # TARGET SELECTION
    # ---------------------------------------------------
    target_column = st.selectbox(
        "Select Target Column",
        df.columns,
        key="target_select"
    )

    # Features and target
    X = df.drop(columns=[target_column])

    # Convert B/M to 0/1 (same as training)
    y = df[target_column].map({'B': 0, 'M': 1})

    # ---------------------------------------------------
    # MODEL SELECTION
    # ---------------------------------------------------
    st.subheader("Select Classification Model")

    model_option = st.selectbox(
        "Choose Model",
        (
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Gaussian Naive Bayes",
            "Random Forest",
            "XGBoost"
        ),
        key="model_select"
    )

    # ---------------------------------------------------
    # RUN MODEL
    # ---------------------------------------------------
    if st.button("Run Selected Model"):

        # Load model (pipeline already saved)
        if model_option == "Logistic Regression":
            model = joblib.load(r"models/lr_2025ab05262.pkl")

        elif model_option == "Decision Tree":
            model = joblib.load(r"models/dt_2025ab05262.pkl")

        elif model_option == "KNN":
            model = joblib.load(r"models/knn_2025ab05262.pkl")

        elif model_option == "Gaussian Naive Bayes":
            model = joblib.load(r"models/gnb_2025ab05262.pkl")

        elif model_option == "Random Forest":
            model = joblib.load(r"models/rf_2025ab05262.pkl")

        elif model_option == "XGBoost":
            model = joblib.load(r"models/xgb_2025ab05262.pkl")

        # Ensure column order matches training
        X = X[model.feature_names_in_]

        # Predictions
        y_pred = model.predict(X)

        # AUC (if available)
        try:
            y_proba = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, y_proba)
        except:
            auc = None

        # ---------------------------------------------------
        # METRICS
        # ---------------------------------------------------
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)

        st.subheader("Model Evaluation Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", round(acc, 4))
        col2.metric("Precision", round(prec, 4))
        col3.metric("Recall", round(rec, 4))

        col4, col5, col6 = st.columns(3)
        col4.metric("F1 Score", round(f1, 4))
        col5.metric("MCC", round(mcc, 4))
        col6.metric("AUC", round(auc, 4) if auc is not None else "N/A")

        # ---------------------------------------------------
        # CONFUSION MATRIX
        # ---------------------------------------------------
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)

        # ---------------------------------------------------
        # CLASSIFICATION REPORT
        # ---------------------------------------------------
        st.subheader("Classification Report")
        st.text(classification_report(y, y_pred))

        st.success("Model execution completed successfully!")
