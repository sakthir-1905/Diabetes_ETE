import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(layout="wide")

# Load
pd.StringDtype = pd.StringDtype
model = joblib.load("best_model.pkl")
threshold = joblib.load("threshold.pkl")

results_df = pd.read_csv("model_results.csv")

st.title("🩺 Diabetes Prediction Dashboard")

# Sidebar
page = st.sidebar.radio("Navigation", ["Model Comparison", "Visualization", "Prediction"])

# -----------------------
# MODEL COMPARISON
# -----------------------
if page == "Model Comparison":
    st.dataframe(results_df)
    best_model = results_df.iloc[0]["model"]
    st.success(f"Best Model: {best_model}")

# -----------------------
# VISUALIZATION
# -----------------------
elif page == "Visualization":
    metric = st.selectbox("Metric", ["accuracy", "precision", "recall", "f1", "roc_auc"])

    fig = px.bar(results_df, x="model", y=metric, color=metric)
    st.plotly_chart(fig)

# -----------------------
# PREDICTION
# -----------------------
else:
    st.subheader("Enter Patient Details")

    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 1, 100)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    smoking = st.selectbox("Smoking History", ["never", "former", "current"])
    bmi = st.number_input("BMI", 10.0, 50.0)
    HbA1c = st.number_input("HbA1c", 3.0, 15.0)
    glucose = st.number_input("Glucose", 50, 300)

    if st.button("Predict"):
        input_data = pd.DataFrame({
            "gender": [gender],
            "age": [age],
            "hypertension": [hypertension],
            "heart_disease": [heart_disease],
            "smoking_history": [smoking],
            "bmi": [bmi],
            "HbA1c_level": [HbA1c],
            "blood_glucose_level": [glucose]
        })

        prob = model.predict_proba(input_data)[0][1]
        pred = 1 if prob >= threshold else 0

        st.write(f"Probability: {prob:.2f}")

        if pred == 1:
            st.error("⚠️ High Risk of Diabetes")
        else:
            st.success("✅ Low Risk")