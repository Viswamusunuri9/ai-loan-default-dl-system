import streamlit as st
import numpy as np
from src.predict import load_trained_model, predict

st.set_page_config(page_title="Loan Default Prediction", layout="centered")

st.title("🏦 AI Loan Default Prediction System")

st.write("Predict the risk of loan default based on applicant profile.")

# Load model
@st.cache_resource
def get_model():
    return load_trained_model()

model = get_model()

# ---------------- INPUT SECTION ---------------- #

st.subheader("📋 Applicant Details")

income = st.slider("Annual Income", 10000, 1000000, 50000)
loan_amount = st.slider("Loan Amount", 1000, 500000, 20000)
credit_score = st.slider("Credit Score", 300, 900, 650)
employment_years = st.slider("Years of Employment", 0, 30, 5)

# ---------------- FEATURE CONVERSION ---------------- #
# NOTE: mapping few inputs into 120-dim vector (demo strategy)

def create_input_vector():
    vec = np.random.rand(120) * 0.1  # small noise

    vec[0] = income / 100000
    vec[1] = loan_amount / 100000
    vec[2] = credit_score / 900
    vec[3] = employment_years / 30

    return vec.reshape(1, -1)

# ---------------- PREDICTION ---------------- #

st.subheader("📊 Prediction")

if st.button("Predict Risk"):

    input_data = create_input_vector()

    prob, label = predict(model, input_data)

    st.metric("Default Probability", f"{prob:.2f}")

    if prob > 0.3:
        st.error("🔴 High Risk - Likely to Default")
    elif prob > 0.15:
        st.warning("🟡 Medium Risk - Caution")
    else:
        st.success("🟢 Low Risk - Safe")

# ---------------- FOOTER ---------------- #

st.caption("⚠️ Demo system: simplified input mapping for visualization.")