# Importing libraries
import numpy as np
import streamlit as st
import joblib

# Loading files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Page configurations
st.set_page_config(page_title="Diabetes Predictions")
st.title("Diabetes Prediction App")
st.write("Use this prediction tool to find diabetes")
image = st.image("image_diabetes.png")

# Inputs
Gender = st.selectbox("Gender",["male","female"])
Age=st.number_input("Enter your Age",min_value=0,max_value=120,step=1)
Bp=st.selectbox("Have hypertension",["no","yes"])
Heart_disease=st.selectbox("History of heartdisease",["no","yes"])
Smoking = st.radio("Smoking History",["No","Not willing to disclose","Quitted","Yes"])
c_bmi = st.checkbox("Do you Want to calculate your BMI")
if c_bmi:
    w = st.sidebar.number_input("Please Enter Your Weight in kg",min_value=1)
    h = st.sidebar.number_input("Please Enter Your Height in cm",min_value=1,step=1)
    if st.sidebar.button("Calculate"):
        b = w/(np.pow(h/100,2))
        if b<=25:
            st.sidebar.success(f"Your BMI value is {b:.2f}")
        else:
            st.sidebar.error(f"Your BMI value is {b:.2f}")
Bmi = st.number_input("Enter BMI",min_value=0.0,max_value = 100.0)
HbA1c = st.number_input("Enter HbA1c_level",min_value=1.0,max_value = 14.0)
Glucose = st.number_input("Enter Blood Glucose Level",step=1)

# Encode inputs
binary_map = {"no": 0, "yes": 1,"female" : 0,"male" : 1}
smoke_map = {
    "No": 0,
    "Not willing to disclose": 1,
    "Quitted": 2,
    "Yes" : 3
}
input_data = np.array([[
    binary_map[Gender],
    Age,
    binary_map[Bp],
    binary_map[Heart_disease],
    smoke_map[Smoking],
    Bmi,
    HbA1c,
    Glucose
]])

#Scale input
input_scaled = scaler.transform(input_data)
#predict output
if st.button("Check for Diabetes"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("You Have Diabetes...!\nConsult a Doctor Immediately")
    else:
        st.success("---You Don't Have Diabetes--- \n Keep a Healthy Routine to stay safe")
