import streamlit as st
import pandas as pd
import pickle


# Model Loader
@st.cache_resource
def load_pipeline():
    try:
        with open("loan_pipeline.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file not found.")
        st.stop()


# Feature Preparation
def prepare_input(user_data, model_columns):
    df = pd.DataFrame([user_data])

    nums = [
        "Applicant_ID", "Applicant_Income", "Coapplicant_Income",
        "Age", "Dependents", "Credit_Score", "Existing_Loans",
        "DTI_Ratio", "Savings", "Collateral_Value",
        "Loan_Amount", "Loan_Term"
    ]

    cats = [
        "Employment_Status", "Marital_Status", "Loan_Purpose",
        "Property_Area", "Education_Level", "Gender",
        "Employer_Category"
    ]

    num_df = df[nums]
    cat_df = pd.get_dummies(df[cats], drop_first=False)

    final_df = pd.concat([num_df, cat_df], axis=1)

    for col in model_columns:
        if col not in final_df.columns:
            final_df[col] = 0

    return final_df[model_columns]


# Prediction
def predict(pipeline, data):
    try:
        cols = pipeline.feature_names_in_
        df = prepare_input(data, cols)
    except:
        df = pd.DataFrame([data])

    pred = pipeline.predict(df)[0]
    prob = pipeline.predict_proba(df)[0][1]

    return pred, prob


# UI

def main():
    st.set_page_config(
        page_title="Loan Risk Assessment",
        page_icon="📊",
        layout="wide"
    )

    # ----------------- Custom Theme -----------------
    st.markdown("""
    <style>
    .stApp {
        background: #0b1220;
        color: #e5e7eb;
        font-family: Inter, sans-serif;
    }

    .app-title {
        font-size: 2.6rem;
        font-weight: 800;
        text-align: center;
        color: #f8fafc;
        margin-bottom: 0.3rem;
    }

    .app-subtitle {
        text-align: center;
        color: #94a3b8;
        margin-bottom: 2rem;
    }

    .card {
        background: #111827;
        border: 1px solid #1f2933;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }

    .card h3 {
        margin-bottom: 1rem;
        color: #93c5fd;
        font-size: 1.2rem;
    }

    .stButton > button {
        background: #2563eb;
        border-radius: 12px;
        height: 3.2rem;
        font-size: 1.1rem;
        font-weight: 600;
        color: white;
        border: none;
    }

    .approve {
        background: #052e1a;
        border-left: 6px solid #22c55e;
        padding: 1.5rem;
        border-radius: 14px;
    }

    .reject {
        background: #2a0a0a;
        border-left: 6px solid #ef4444;
        padding: 1.5rem;
        border-radius: 14px;
    }

    .metric-box {
        background: #020617;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #1e293b;
    }
    </style>
    """, unsafe_allow_html=True)

    # ----------------- Header -----------------
    st.markdown("<div class='app-title'>Loan Risk Assessment System</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='app-subtitle'>Machine learning based credit evaluation with probability scoring</div>",
        unsafe_allow_html=True
    )

    pipeline = load_pipeline()

    tab1, tab2 = st.tabs(["📝 Application", "ℹ️ Overview"])

    # =====================================================
    # Application Tab
    # =====================================================
    with tab1:
        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("<div class='card'><h3>Applicant Profile</h3>", unsafe_allow_html=True)

            applicant_id = st.number_input("Applicant ID", min_value=1, value=1)
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital = st.selectbox("Marital Status", ["Single", "Married"])
            dependents = st.number_input("Dependents", 0, 10, 0)
            education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='card'><h3>Employment</h3>", unsafe_allow_html=True)

            employment = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed"])
            employer = st.selectbox("Employer Category", ["Private", "Government", "MNC", "Unemployed"])
            income = st.number_input("Monthly Income", value=5000.0)
            co_income = st.number_input("Co-applicant Income", value=0.0)

            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown("<div class='card'><h3>Financial Details</h3>", unsafe_allow_html=True)

            credit = st.number_input("Credit Score", 300, 850, 650)
            loans = st.number_input("Existing Loans", 0, 20, 0)
            dti = st.number_input("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
            savings = st.number_input("Savings", value=10000.0)
            collateral = st.number_input("Collateral Value", value=20000.0)

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='card'><h3>Loan Request</h3>", unsafe_allow_html=True)

            amount = st.number_input("Loan Amount", value=10000.0)
            term = st.number_input("Loan Term (months)", value=36)
            purpose = st.selectbox("Loan Purpose", ["Personal", "Business", "Car", "Home", "Education"])
            area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

            st.markdown("</div>", unsafe_allow_html=True)

        data = {
            "Applicant_ID": applicant_id,
            "Applicant_Income": income,
            "Coapplicant_Income": co_income,
            "Age": age,
            "Dependents": dependents,
            "Credit_Score": credit,
            "Existing_Loans": loans,
            "DTI_Ratio": dti,
            "Savings": savings,
            "Collateral_Value": collateral,
            "Loan_Amount": amount,
            "Loan_Term": term,
            "Employment_Status": employment,
            "Marital_Status": marital,
            "Loan_Purpose": purpose,
            "Property_Area": area,
            "Education_Level": education,
            "Gender": gender,
            "Employer_Category": employer
        }

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Evaluate Risk", use_container_width=True):
            with st.spinner("Evaluating application risk..."):
                pred, prob = predict(pipeline, data)
                pct = prob * 100

                if pct >= 70:
                    st.markdown(
                        f"<div class='approve'><h2>Approved</h2><p>{pct:.1f}% approval probability</p></div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div class='reject'><h2>Not Approved</h2><p>{pct:.1f}% approval probability</p></div>",
                        unsafe_allow_html=True
                    )

                m1, m2, m3 = st.columns(3)
                m1.markdown(f"<div class='metric-box'>Probability<br><b>{pct:.1f}%</b></div>", unsafe_allow_html=True)
                m2.markdown(
                    f"<div class='metric-box'>Risk Level<br><b>{'Low' if pct>=75 else 'Medium' if pct>=55 else 'High'}</b></div>",
                    unsafe_allow_html=True
                )
                m3.markdown("<div class='metric-box'>Threshold<br><b>70%</b></div>", unsafe_allow_html=True)

                st.progress(prob)

   
    # About
    with tab2:
        st.markdown("""
        ### Project Overview

        This system evaluates loan eligibility using supervised machine learning.
        It combines applicant demographics, financial health, and loan attributes
        to estimate approval probability.

        **Why this project stands out**
        - End-to-end ML pipeline
        - Feature-aligned preprocessing
        - Clean, production-style UI
        - Resume-ready fintech use case
        """)

if __name__ == "__main__":
    main()
