import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("lung_cancer_model.pkl", "rb"))

st.title("🫁 Lung Cancer Prediction App")

age = st.number_input("Age", min_value=1, max_value=120, value=40)
smoking = st.selectbox("Smoking", ["Yes", "No"])
yellow_fingers = st.selectbox("Yellow Fingers", ["Yes", "No"])
anxiety = st.selectbox("Anxiety", ["Yes", "No"])
chronic_disease = st.selectbox("Chronic Disease", ["Yes", "No"])

data = np.array([
    age,
    1 if smoking == "Yes" else 0,
    1 if yellow_fingers == "Yes" else 0,
    1 if anxiety == "Yes" else 0,
    1 if chronic_disease == "Yes" else 0
]).reshape(1, -1)

if st.button("Predict"):
    pred = model.predict(data)
    if pred[0] == 1:
        st.error("⚠️ High risk of Lung Cancer!")
    else:
        st.success("✅ Low risk of Lung Cancer.")
