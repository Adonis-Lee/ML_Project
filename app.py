import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Sayfa Ayarları
st.set_page_config(page_title="CENG465 ML Toolkit", layout="wide", page_icon="🤖")

st.title("🎓 CENG465 - Group ML Toolkit")
st.markdown("This tool allows you to Train, Test and Evaluate ML models on any CSV dataset.")

# --- SOL PANEL (Ayarlar) ---
st.sidebar.header("1. Upload & Settings")
uploaded = st.sidebar.file_uploader("Upload your Dataset (CSV)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    # --- ORTA ALAN (Veri Önizleme) ---
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # --- AYARLAR ---
    st.sidebar.subheader("2. Preprocessing")

    # Hedef Sütun Seçimi
    cols = df.columns.tolist()
    target = st.sidebar.selectbox("Select Target Column (Class Label)", cols)

    # Preprocessing Seçenekleri
    scaler_choice = st.sidebar.selectbox("Normalization Method", ["None", "StandardScaler", "MinMaxScaler"])
    encoding_choice = st.sidebar.checkbox("Apply One-Hot Encoding (Auto-detect categorical)", value=True)

    # Train/Test Split
    st.sidebar.subheader("3. Split & Model")
    test_size = st.sidebar.slider("Test Set Ratio", 0.1, 0.5, 0.3)

    # Model Seçimi
    model_name = st.sidebar.selectbox("Select Classifier", 
                                      ["Perceptron", 
                                       "Multilayer Perceptron (Backprop)", 
                                       "Decision Tree"])

    # --- ÇALIŞTIR BUTONU ---
    if st.sidebar.button("🚀 Train Model"):

        # 1. Veri Hazırlığı
        X = df.drop(columns=[target])
        y = df[target]

        # Target (y) eğer yazı ise (Örn: Pass/Fail), LabelEncoder ile sayıya (0/1) çeviriyoruz
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            st.info(f"Target column encoded: {le.classes_}")

        # One-Hot Encoding (Otomatik - PDF Şartı)
        if encoding_choice:
            # Sadece kategorik (object) sütunları bul ve encode et
            X = pd.get_dummies(X, drop_first=True)
            st.write(f"dataset shape after encoding: {X.shape}")

        # Normalization (PDF Şartı)
        if scaler_choice == "StandardScaler":
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        elif scaler_choice == "MinMaxScaler":
            scaler = MinMaxScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # 2. Model Eğitimi
        if model_name == "Perceptron":
            model = Perceptron()
        elif model_name == "Multilayer Perceptron (Backprop)":
            model = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=1000, random_state=42)
        else:
            model = DecisionTreeClassifier(criterion='entropy', random_state=42)

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # --- SONUÇ EKRANI (Tablar ile Düzenli Görünüm) ---
            st.divider()
            st.header(f"Results for: {model_name}")

            tab1, tab2, tab3 = st.tabs(["📈 Metrics", "🟦 Confusion Matrix", "🌳 Model Details"])

            with tab1:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
                col2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
                col3.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
                col4.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")

                st.text("Detailed Classification Report:")
                st.code(classification_report(y_test, y_pred, zero_division=0))

            with tab2:
                st.write("Confusion Matrix Heatmap")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)

            with tab3:
                st.write("Model Parameters:")
                st.json(model.get_params())

                if model_name == "Decision Tree":
                    st.write("Decision Tree Visualization:")
                    fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
                    plot_tree(model, filled=True, feature_names=X.columns, ax=ax_tree, max_depth=3)
                    st.pyplot(fig_tree)

        except Exception as e:
            st.error(f"An error occurred during training: {e}")
            st.warning("Hint: Check if your dataset contains unprocessable text data.")

else:
    st.info("Please upload a CSV file from the sidebar to begin.")
