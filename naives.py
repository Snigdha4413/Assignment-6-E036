import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration for a professional layout
st.set_page_config(page_title="DataScience Studio", layout="wide")

st.title("🚀 DataScience Studio: Auto-ML & Analytics")
st.markdown("Upload your dataset to explore patterns and deploy machine learning models instantly.")

# 1. Input Dataset Field
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Create Tabs for the main interface
    tab_eda, tab_ml = st.tabs(["📊 Exploratory Analysis", "🤖 ML Model Development"])

    # --- TAB 1: EXPLORATORY ANALYSIS ---
    with tab_eda:
        st.subheader("📈 General Dataset Analytics")
        
        # Summary Metrics
        ana_col1, ana_col2, ana_col3, ana_col4 = st.columns(4)
        ana_col1.metric("Total Rows", df.shape[0])
        ana_col2.metric("Total Columns", df.shape[1])
        ana_col3.metric("Missing Values", df.isnull().sum().sum())
        ana_col4.metric("Duplicate Rows", df.duplicated().sum())

        st.divider()
        
        # Visualizations for EDA
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.write("#### Data Distribution (Numerical)")
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                selected_viz = st.selectbox("Select column to visualize", num_cols)
                fig_dist, ax_dist = plt.subplots(figsize=(8, 4))
                sns.histplot(df[selected_viz], kde=True, color="skyblue", ax=ax_dist)
                st.pyplot(fig_dist)
            else:
                st.info("No numerical columns found for distribution plots.")

        with viz_col2:
            st.write("#### Correlation Heatmap")
            if len(num_cols) > 1:
                fig_corr, ax_corr = plt.subplots(figsize=(8, 4))
                sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
                st.pyplot(fig_corr)
            else:
                st.info("Need at least two numerical columns for a heatmap.")

        st.write("#### Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

    # --- TAB 2: ML MODEL DEVELOPMENT ---
    with tab_ml:
        col_settings, col_results = st.columns([1, 2])
        
        with col_settings:
            st.subheader("⚙️ Model Settings")
            
            # Model Selection
            selected_model = st.selectbox("Choose Algorithm", ["Naive Bayes", "Linear Regression"])
            
            # Task Type Logic
            task_type = st.radio("Select Task Type", ["Classification", "Regression"])
            
            if task_type == "Classification":
                target_options = [col for col in df.columns if df[col].nunique() < 20 or df[col].dtype == 'object']
                st.info("Discrete/Categorical targets filtered for Classification.")
            else:
                target_options = [col for col in df.columns if np.issubdtype(df[col].dtype, np.number)]
                st.info("Continuous numeric targets filtered for Regression.")

            target_field = st.selectbox("Select Target Field", target_options)
            
            # Feature selection
            feature_columns = [col for col in df.columns if col != target_field]
            selected_features = st.multiselect("Select Feature Fields", feature_columns, default=feature_columns)
            
            test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100
            run_btn = st.button("🚀 Train & Evaluate")

        with col_results:
            st.subheader("🎯 Training Results")
            if run_btn:
                if not selected_features:
                    st.error("Please select at least one feature.")
                else:
                    try:
                        # --- DATA PREPARATION ---
                        X = pd.get_dummies(df[selected_features])
                        y = df[target_field]
                        
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42
                        )

                        if task_type == "Classification":
                            # Fix for 'Unknown label type'
                            y_train_c = y_train.astype(str)
                            y_test_c = y_test.astype(str)
                            
                            model = GaussianNB()
                            model.fit(X_train, y_train_c)
                            
                            y_train_pred = model.predict(X_train)
                            y_test_pred = model.predict(X_test)
                            
                            # Metrics
                            m_col1, m_col2 = st.columns(2)
                            m_col1.metric("Train Accuracy", f"{accuracy_score(y_train_c, y_train_pred):.2%}")
                            m_col2.metric("Test Accuracy", f"{accuracy_score(y_test_c, y_test_pred):.2%}")

                            # Confusion Matrices
                            st.write("#### Confusion Matrices (Train vs Test)")
                            cm_c1, cm_c2 = st.columns(2)
                            labels = np.unique(np.concatenate((y_train_c, y_test_c)))

                            with cm_c1:
                                fig_tr, ax_tr = plt.subplots(figsize=(4, 3))
                                sns.heatmap(confusion_matrix(y_train_c, y_train_pred), annot=True, fmt='d', cmap='Greens', cbar=False, xticklabels=labels, yticklabels=labels)
                                plt.title("Training Set", fontsize=10)
                                st.pyplot(fig_tr)

                            with cm_c2:
                                fig_ts, ax_ts = plt.subplots(figsize=(4, 3))
                                sns.heatmap(confusion_matrix(y_test_c, y_test_pred), annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels)
                                plt.title("Testing Set", fontsize=10)
                                st.pyplot(fig_ts)

                        else:
                            # REGRESSION LOGIC
                            model = LinearRegression()
                            model.fit(X_train, y_train)
                            
                            y_test_pred = model.predict(X_test)
                            st.metric("Model R² Score", f"{r2_score(y_test, y_test_pred):.4f}")
                            
                            fig_reg, ax_reg = plt.subplots(figsize=(8, 4))
                            plt.scatter(y_test, y_test_pred, alpha=0.5, color='orange')
                            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
                            plt.xlabel("Actual")
                            plt.ylabel("Predicted")
                            st.pyplot(fig_reg)

                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.info("Configure settings and click 'Train' to see results.")

else:
    st.info("Waiting for CSV upload to begin analysis.")