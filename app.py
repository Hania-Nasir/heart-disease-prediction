import streamlit as st
import numpy as np
import joblib
import os

model=joblib.load(os.path.join("model_catboost_heart_disease.joblib"))

st.set_page_config("Heart Disease Predictor")

st.title("Heart Disease Prediction app")
st.write("enter the following parameters to predict you have a heart disease or not")

st.divider()

col1,col2=st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)

chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)

thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)

oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)

ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)

with col2:
    sex = st.selectbox("Sex",options={"female":0,"male":1})

    cp = st.selectbox("Chest Pain Type", options={
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
    })
    
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options={"True": 1, "False": 0})

    restecg = st.selectbox("Resting ECG Results", options={
    "Normal": 0,
    "ST-T abnormality": 1,
    "Left ventricular hypertrophy": 2
    })

    exang = st.selectbox("Exercise Induced Angina", options={"Yes": 1, "No": 0})

    slope = st.selectbox("Slope of ST Segment", options={
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
    })

    thal = st.selectbox("Thalassemia", options={
    "Normal": 0,
    "Fixed Defect": 1,
    "Reversible Defect": 2
    })

if st.button("Predict"):
    X_input=np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                       thalach, exang, oldpeak, slope, ca, thal]])

    Prediction=int(model.predict(X_input)[0])

    st.subheader("Prediction Result")
    if Prediction == 1:
        st.write("Prediction:Heart Disease Detected")
    else:
        st.write("Prediction:No Heart Disease Detected")