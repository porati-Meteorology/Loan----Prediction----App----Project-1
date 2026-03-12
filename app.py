import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# 1. Page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Load the saved model and feature column blueprint
# @st.cache_resource loads them once and keeps them in memory
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    with open("feature_columns.json") as f:
        feature_cols = json.load(f)
    return model, feature_cols

try:
    model, feature_columns = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError:
    artifacts_loaded = False

# 3. Reference lists - mirrors the unique values from the training dataset
PROFESSION_LIST = sorted([
    "Mechanical_engineer", "Software_Developer", "Technical_writer",
    "Civil_servant", "Librarian", "Economist", "Flight_attendant",
    "Architect", "Computer_hardware_engineer", "Financial_Analyst",
    "Air_traffic_controller", "Comedian", "Graphic_Designer",
    "Chemical_engineer", "Biomedical_Engineer", "Artist",
    "Magistrate", "Lawyer", "Firefighter", "Surgeon",
    "Dentist", "Physician", "Nurse", "Pharmacist",
    "Statistician", "Analyst", "Chartered_Accountant",
    "Designer", "Drafter", "Technician", "Web_designer",
    "Geologist", "Microbiologist", "Petroleum_Engineer",
    "Industrial_Engineer", "Environmental_Engineer",
    "Hotel_Manager", "Travel_Agent", "Secretary", "Consultant",
    "Police_officer", "Army_officer",
])

STATE_LIST = sorted([
    "Andhra_Pradesh", "Assam", "Bihar", "Chandigarh", "Chhattisgarh",
    "Delhi", "Goa", "Gujarat", "Haryana", "Himachal_Pradesh",
    "Jammu_and_Kashmir", "Jharkhand", "Karnataka", "Kerala",
    "Madhya_Pradesh", "Maharashtra", "Manipur", "Meghalaya",
    "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan",
    "Sikkim", "Tamil_Nadu", "Telangana", "Tripura",
    "Uttar_Pradesh", "Uttarakhand", "West_Bengal",
])

HOUSE_LIST = ["rented", "owned", "norent_noown"]

# 4. Feature engineering function
# Applies the same transformations as the notebook so the input matches
# what the model was trained on
def engineer_features(raw):
    income            = raw["Income"]
    age               = raw["Age"]
    experience        = raw["Experience"]
    married           = raw["Married/Single"]
    house_ownership   = raw["House_Ownership"]
    car_ownership     = raw["Car_Ownership"]
    profession        = raw["Profession"]
    state             = raw["STATE"]
    current_job_yrs   = raw["CURRENT_JOB_YRS"]
    current_house_yrs = raw["CURRENT_HOUSE_YRS"]

    # Age binning - same bins used in notebook Phase 3
    Age_bins   = [21, 30, 55, 100]
    Age_labels = ["Youth(21-30)", "Middle_Aged(31-55)", "Senior(55+)"]
    age_cat = pd.cut([age], bins=Age_bins, labels=Age_labels, right=False)[0]

    # Experience binning - same bins used in notebook Phase 3
    Exp_bins   = [0, 2, 5, 10, float("inf")]
    Exp_labels = ["Junior(0-2)", "Intermediate(2-5)", "Mid-Level(5-10)", "Senior(10+)"]
    exp_level  = pd.cut([experience], bins=Exp_bins, labels=Exp_labels, right=False)[0]

    # Binary mappings
    marital_status = 1 if married == "married" else 0
    car_owner      = 1 if car_ownership == "yes" else 0
    res_stability  = 1 if current_house_yrs > 5 else 0

    # Build the input row with the 11 selected features
    row = {
        "Income":                      income,
        "House_Ownership":             house_ownership,
        "CURRENT_JOB_YRS":             current_job_yrs,
        "Profession":                  profession,
        "STATE":                       state,
        "CURRENT_HOUSE_YRS":           current_house_yrs,
        "Age_Category":                str(age_cat),
        "Experience_Level":            str(exp_level),
        "Marital_Status":              marital_status,
        "Car_Owner":                   car_owner,
        "Residential_Stability_Ratio": res_stability,
    }

    df = pd.DataFrame([row])

    # One-Hot Encode categorical columns
    cat_cols = ["House_Ownership", "Profession", "STATE", "Age_Category", "Experience_Level"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Align to the 91-column training blueprint - fill any missing columns with 0
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df

# 5. Page header
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown("# 🏦")
with col_title:
    st.title("Alpha Dreamers Banking Consortium")
    st.markdown("#### Personal Loan Default Risk Predictor")

st.markdown("---")

if not artifacts_loaded:
    st.error("Model files not found. Place model.pkl and feature_columns.json in the same folder as app.py")
    st.stop()

# 6. Sidebar form - loan officer enters applicant details here
st.sidebar.header("📋 Applicant Information")
st.sidebar.markdown("Fill in the applicant details below :")

with st.sidebar:
    income = st.number_input("Annual Income (Rs)", min_value=10_000, max_value=10_000_000, value=500_000, step=10_000)
    age = st.slider("Age", min_value=21, max_value=80, value=35)
    experience = st.slider("Years of Work Experience", min_value=0, max_value=40, value=5)
    married = st.selectbox("Marital Status", options=["single", "married"], format_func=lambda x: x.capitalize())
    house_ownership = st.selectbox("House Ownership", options=HOUSE_LIST, format_func=lambda x: x.replace("_", " ").title())
    car_ownership = st.selectbox("Car Ownership", options=["no", "yes"], format_func=lambda x: "Yes" if x == "yes" else "No")
    profession = st.selectbox("Profession", options=PROFESSION_LIST)
    state = st.selectbox("State of Residence", options=STATE_LIST, format_func=lambda x: x.replace("_", " "))
    current_job_yrs = st.slider("Years in Current Job", min_value=0, max_value=20, value=3)
    current_house_yrs = st.slider("Years in Current Residence", min_value=0, max_value=20, value=8)
    predict_btn = st.button("Predict Default Risk", use_container_width=True)

# 7. Main display - applicant summary and prediction result side by side
col_info, col_result = st.columns([1, 1], gap="large")

with col_info:
    st.subheader("📊 Applicant Summary")
    summary = {
        "Field": ["Annual Income", "Age", "Work Experience", "Marital Status",
                  "House Ownership", "Car Ownership", "Profession", "State",
                  "Current Job (yrs)", "Current Residence (yrs)"],
        "Value": [
            f"Rs {income:,}", age, f"{experience} yrs", married.capitalize(),
            house_ownership.replace("_", " ").title(),
            "Yes" if car_ownership == "yes" else "No",
            profession.replace("_", " "), state.replace("_", " "),
            current_job_yrs, current_house_yrs
        ]
    }
    st.dataframe(pd.DataFrame(summary).set_index("Field"), use_container_width=True, height=390)

with col_result:
    st.subheader("🎯 Prediction Result")

    if predict_btn:
        raw_input = {
            "Income": income, "Age": age, "Experience": experience,
            "Married/Single": married, "House_Ownership": house_ownership,
            "Car_Ownership": car_ownership, "Profession": profession,
            "STATE": state, "CURRENT_JOB_YRS": current_job_yrs,
            "CURRENT_HOUSE_YRS": current_house_yrs,
        }

        processed_df    = engineer_features(raw_input)
        prediction      = model.predict(processed_df)[0]
        probability     = model.predict_proba(processed_df)[0]
        default_prob    = probability[1] * 100
        no_default_prob = probability[0] * 100

        if prediction == 1:
            st.error("### ⚠️ HIGH DEFAULT RISK")
            st.markdown(f"This applicant is predicted to **DEFAULT** on their loan.  \nDefault probability : **{default_prob:.1f}%**")
        else:
            st.success("### ✅ LOW DEFAULT RISK")
            st.markdown(f"This applicant is predicted to **repay** their loan.  \nDefault probability : **{default_prob:.1f}%**")

        st.markdown("#### Default Probability Breakdown")
        g1, g2 = st.columns(2)
        with g1:
            st.metric(label="⚠️ Default Probability",  value=f"{default_prob:.1f}%")
        with g2:
            st.metric(label="✅ Repayment Probability", value=f"{no_default_prob:.1f}%")

        st.progress(int(default_prob))

        st.markdown("---")
        st.markdown("#### 📝 Loan Officer Recommendation")
        if default_prob < 20:
            st.success("**APPROVE** - Very low risk. Proceed with standard loan terms.")
        elif default_prob < 40:
            st.warning("**REVIEW** - Moderate risk. Request additional documents or a co-signer.")
        elif default_prob < 60:
            st.warning("**CAUTION** - Elevated risk. Consider a smaller loan amount or higher interest rate.")
        else:
            st.error("**DECLINE** - High probability of default. Do not approve this application.")

    else:
        st.info("Fill in the applicant details in the sidebar and click Predict Default Risk to see the result.")

# 8. Footer
st.markdown("---")
st.markdown("<small>Alpha Dreamers Banking Consortium · Business Intelligence Division · Logistic Regression (scikit-learn) · Validation Accuracy : 87.8%</small>", unsafe_allow_html=True)