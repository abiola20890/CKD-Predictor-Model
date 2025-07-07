# ckd_app.py
import streamlit as st
import numpy as np
import joblib
import datetime

# Load the CKD model
model = joblib.load("kidney_xgboost_model.pkl")


# App title and banner
st.title("🩺 Chronic Kidney Disease (CKD) Prediction")
st.image("kidney_banner.png", use_column_width=True)
st.write("## By DSA 2025")

# Instructions
st.markdown("""
👉 Please use the **sidebar** to enter your medical information and predict the likelihood of Chronic Kidney Disease.
""")

# Sidebar Inputs
st.sidebar.header("🔍 Enter Patient Details")
sg = st.sidebar.number_input("Specific Gravity (sg)", min_value=1.000, max_value=1.030, step=0.001, format="%.3f")
al = st.sidebar.number_input("Albumin (al)", min_value=0, step=1)
su = st.sidebar.number_input("Sugar (su)", min_value=0, step=1)
rbc = st.sidebar.selectbox("Red Blood Cells (rbc)", options=["normal", "abnormal"])
pc = st.sidebar.selectbox("Pus Cell (pc)", options=["normal", "abnormal"])
bu = st.sidebar.number_input("Blood Urea (bu)", min_value=0.0, step=0.1)
sc = st.sidebar.number_input("Serum Creatinine (sc)", min_value=0.0, step=0.1)
hemo = st.sidebar.number_input("Hemoglobin (hemo)", min_value=0.0, step=0.1)
pcv = st.sidebar.number_input("Packed Cell Volume (pcv)", min_value=0, step=1)
rc = st.sidebar.number_input("Red Blood Cell Count (rc)", min_value=0.0, step=0.01)
htn = st.sidebar.selectbox("Hypertension (htn)", options=["yes", "no"])
dm = st.sidebar.selectbox("Diabetes Mellitus (dm)", options=["yes", "no"])
appet = st.sidebar.selectbox("Appetite (appet)", options=["good", "poor"])
pe = st.sidebar.selectbox("Pedal Edema (pe)", options=["yes", "no"])

# Encode categorical features
rbc_encoded = 1 if rbc == "abnormal" else 0
pc_encoded = 1 if pc == "abnormal" else 0
htn_encoded = 1 if htn == "yes" else 0
dm_encoded = 1 if dm == "yes" else 0
appet_encoded = 1 if appet == "poor" else 0
pe_encoded = 1 if pe == "yes" else 0

# Prediction button and logic
if st.button("Predict"):
    input_data = np.array([[sg, al, su, rbc_encoded, pc_encoded,
                            bu, sc, hemo, pcv, rc,
                            htn_encoded, dm_encoded, appet_encoded, pe_encoded]])
    
    prediction = model.predict(input_data)[0]
    result = "🟢 No CKD" if prediction == 0 else "🔴 CKD Detected"
    
   # Get probability
    proba = model.predict_proba(input_data)[0][1]  # Probability of CKD

    st.markdown("##")
    st.subheader("Prediction Result:")
    st.success(result)
    st.metric("🔢 CKD Probability", f"{proba * 100:.2f}%")
    
     # Downloadable report
    report_text = f"""
    Chronic Kidney Disease (CKD) Prediction Report
    ----------------------------------------------
    Result: {result}
    CKD Probability: {proba * 100:.2f}%
    """
    st.download_button(
        label="📄 Download Report",
        data=report_text,
        file_name="ckd_prediction_report.txt",
        mime="text/plain"
    )

    st.markdown("---")
    st.info("📌 **Disclaimer**: This tool is for educational/demo purposes only and not a substitute for professional medical advice. Always consult qualified healthcare providers regarding your medical conditions.")

# Feature explanations
with st.expander("📘 What do these features mean?"):
    st.write("**sg**: Specific Gravity — concentration of urine.")
    st.write("**al**: Albumin — protein levels in urine.")
    st.write("**su**: Sugar — glucose in urine.")
    st.write("**rbc**: Red Blood Cells — normal or abnormal presence.")
    st.write("**pc**: Pus Cells — infection indicator.")
    st.write("**bu**: Blood Urea — waste filtered by kidneys.")
    st.write("**sc**: Serum Creatinine — waste level in blood.")
    st.write("**hemo**: Hemoglobin — red blood cell protein.")
    st.write("**pcv**: Packed Cell Volume — % of blood occupied by cells.")
    st.write("**rc**: Red Blood Cell Count.")
    st.write("**htn**: Hypertension — high blood pressure.")
    st.write("**dm**: Diabetes Mellitus — diabetes history.")
    st.write("**appet**: Appetite — good or poor.")
    st.write("**pe**: Pedal Edema — fluid retention/swelling.")