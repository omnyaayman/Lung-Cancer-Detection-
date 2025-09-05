import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ======================
# Load Model
# ======================
# المسار الصحيح للملف داخل فولدر src
model_path = os.path.join(os.path.dirname(__file__), "lung_cancer_model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# ======================
# App Layout
# ======================
st.set_page_config(
    page_title="Lung Cancer Prediction",
    page_icon="🫁",
    layout="wide"
)

st.title("🫁 Lung Cancer Prediction App")
st.markdown("### A modern tool to assess **lung cancer risk** based on health and lifestyle features.")

st.markdown("---")

# ======================
# Sidebar Inputs
# ======================
st.sidebar.header("🔧 Input Features")

gender = st.sidebar.selectbox("Gender", ["M", "F"])
age = st.sidebar.slider("Age", 20, 100, 40)
smoking = st.sidebar.selectbox("Smoking", [0, 1])
yellow_fingers = st.sidebar.selectbox("Yellow Fingers", [0, 1])
anxiety = st.sidebar.selectbox("Anxiety", [0, 1])
peer_pressure = st.sidebar.selectbox("Peer Pressure", [0, 1])
chronic_disease = st.sidebar.selectbox("Chronic Disease", [0, 1])
fatigue = st.sidebar.selectbox("Fatigue", [0, 1])
allergy = st.sidebar.selectbox("Allergy", [0, 1])
wheezing = st.sidebar.selectbox("Wheezing", [0, 1])
alcohol = st.sidebar.selectbox("Alcohol Consuming", [0, 1])
coughing = st.sidebar.selectbox("Coughing", [0, 1])
shortness_breath = st.sidebar.selectbox("Shortness of Breath", [0, 1])
swallowing_difficulty = st.sidebar.selectbox("Swallowing Difficulty", [0, 1])
chest_pain = st.sidebar.selectbox("Chest Pain", [0, 1])

# Gender encoding
gender_val = 1 if gender == "M" else 0

# ======================
# Prediction
# ======================
features = np.array([[gender_val, age, smoking, yellow_fingers, anxiety,
                     peer_pressure, chronic_disease, fatigue, allergy,
                     wheezing, alcohol, coughing, shortness_breath,
                     swallowing_difficulty, chest_pain]])

if st.sidebar.button("🔍 Predict"):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1] if hasattr(model, "predict_proba") else None

    # Result Card
    st.markdown("## 🎯 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Lung Cancer Detected")
    else:
        st.success("✅ Low Risk of Lung Cancer")

    if probability is not None:
        st.markdown(f"### Probability of Cancer: **{probability:.2%}**")

        # ======================
        # Chart 1: Probability Bar
        # ======================
        st.markdown("#### 📊 Risk Probability Chart")
        prob_df = pd.DataFrame({
            "Risk": ["No Cancer", "Cancer"],
            "Probability": [1 - probability, probability]
        })
        fig, ax = plt.subplots()
        sns.barplot(x="Risk", y="Probability", data=prob_df, palette="coolwarm", ax=ax)
        st.pyplot(fig)

        # ======================
        # Chart 2: Features Importance (if available)
        # ======================
        if hasattr(model, "feature_importances_"):
            st.markdown("#### 🔎 Feature Importance")
            importance = model.feature_importances_
            feature_names = ["Gender","Age","Smoking","Yellow_Fingers","Anxiety","Peer_Pressure",
                             "Chronic_Disease","Fatigue","Allergy","Wheezing","Alcohol","Coughing",
                             "Shortness_Breath","Swallowing_Difficulty","Chest_Pain"]

            imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
            imp_df = imp_df.sort_values("Importance", ascending=False)

            fig2, ax2 = plt.subplots(figsize=(8,5))
            sns.barplot(x="Importance", y="Feature", data=imp_df, palette="viridis", ax=ax2)
            st.pyplot(fig2)
