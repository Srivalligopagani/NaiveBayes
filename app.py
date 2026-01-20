import streamlit as st
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="End-to-End Naive Bayes", layout="wide")

# --------------------------------------------------
# Paths & Session State
# --------------------------------------------------
BASE_DIR = os.getcwd()
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

if "df" not in st.session_state:
    st.session_state.df = None
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("📊 End-to-End Naive Bayes Machine Learning Platform")

# --------------------------------------------------
# Sidebar – Naive Bayes Settings
# --------------------------------------------------
st.sidebar.header("Naive Bayes Settings")

nb_type = st.sidebar.selectbox(
    "Select Naive Bayes Model",
    ["GaussianNB", "MultinomialNB", "BernoulliNB"]
)

# --------------------------------------------------
# STEP 1: DATA INGESTION
# --------------------------------------------------
st.header("Step 1: Data Ingestion")

source = st.radio("Choose Data Source", ["Download Iris Dataset", "Upload CSV"])

if source == "Download Iris Dataset":
    if st.button("Download Dataset"):
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        response = requests.get(url)
        path = os.path.join(RAW_DIR, "iris.csv")

        with open(path, "wb") as f:
            f.write(response.content)

        st.session_state.df = pd.read_csv(path)
        st.success("Iris dataset downloaded successfully")

elif source == "Upload CSV":
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded:
        path = os.path.join(RAW_DIR, uploaded.name)
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())

        st.session_state.df = pd.read_csv(path)
        st.success("CSV uploaded successfully")

# --------------------------------------------------
# STEP 2: EDA
# --------------------------------------------------
if st.session_state.df is not None:
    st.header("Step 2: Exploratory Data Analysis")

    df = st.session_state.df
    st.dataframe(df.head())

    st.write("Shape:", df.shape)
    st.write("Missing Values:")
    st.write(df.isnull().sum())

    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    plt.close(fig)

# --------------------------------------------------
# STEP 3: DATA CLEANING
# --------------------------------------------------
if st.session_state.df is not None:
    st.header("Step 3: Data Cleaning")

    strategy = st.selectbox(
        "Missing Value Strategy",
        ["Mean", "Median", "Drop Rows"]
    )

    df_clean = st.session_state.df.copy()

    if strategy == "Drop Rows":
        df_clean = df_clean.dropna()
    else:
        for col in df_clean.select_dtypes(include=np.number):
            if strategy == "Mean":
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)

    st.session_state.df_clean = df_clean
    st.success("Data cleaning completed")
    st.dataframe(df_clean.head())

# --------------------------------------------------
# STEP 4: SAVE CLEANED DATA
# --------------------------------------------------
st.header("Step 4: Save Cleaned Dataset")

if st.button("Save Cleaned Dataset"):
    if st.session_state.df_clean is None:
        st.error("No cleaned dataset available")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cleaned_dataset_{ts}.csv"
        path = os.path.join(CLEAN_DIR, filename)

        st.session_state.df_clean.to_csv(path, index=False)
        st.success("Cleaned dataset saved")
        st.info(f"Saved at: {path}")

# --------------------------------------------------
# STEP 5: LOAD CLEANED DATA
# --------------------------------------------------
st.header("Step 5: Load Cleaned Dataset")

files = os.listdir(CLEAN_DIR)
if files:
    selected = st.selectbox("Select Dataset", files)
    df_model = pd.read_csv(os.path.join(CLEAN_DIR, selected))
    st.success("Dataset loaded successfully")
    st.dataframe(df_model.head())
else:
    st.warning("No cleaned datasets found")

# --------------------------------------------------
# STEP 6: TRAIN NAIVE BAYES
# --------------------------------------------------
if "df_model" in locals():
    st.header("Step 6: Train Naive Bayes Model")

    target = st.selectbox("Select Target Column", df_model.columns)
    y = df_model[target]

    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    X = df_model.drop(columns=[target])
    X = X.select_dtypes(include=np.number)

    if X.empty:
        st.error("No numeric features available for training")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Model selection
    if nb_type == "GaussianNB":
        model = GaussianNB()
        X_train_final, X_test_final = X_train, X_test

    elif nb_type == "MultinomialNB":
        model = MultinomialNB()
        X_train_final = np.abs(X_train)
        X_test_final = np.abs(X_test)

    else:
        model = BernoulliNB()
        X_train_final = (X_train > 0).astype(int)
        X_test_final = (X_test > 0).astype(int)

    model.fit(X_train_final, y_train)
    y_pred = model.predict(X_test_final)

    acc = accuracy_score(y_test, y_pred)
    st.success(f"Accuracy: {acc:.2f}")

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))
