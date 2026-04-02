import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

# ==========================
# Load Model
# ==========================
@st.cache_resource
def load_model():
    model = joblib.load("models/best_model.pkl")
    threshold = joblib.load("models/optimal_threshold.pkl")
    return model, threshold

model, threshold = load_model()

# ==========================
# App Config
# ==========================
st.set_page_config(page_title="Bank Term Deposit Prediction", layout="wide")

st.title("🏦 Bank Term Deposit Prediction System")
st.markdown("Predict whether a client will subscribe to a term deposit.")

# ==========================
# Sidebar
# ==========================
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to:", ["Manual Prediction", "Bulk Prediction"])

# ==========================
# Feature Columns
# ==========================
columns = [
    'age','job','marital','education','default','balance','housing','loan',
    'contact','day','month','duration','campaign','pdays','previous','poutcome'
]

# ==========================
# Manual Prediction
# ==========================
if option == "Manual Prediction":
    st.header("📝 Client Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 18, 100, 30)
        job = st.selectbox("Job", ['admin.', 'technician', 'services', 'management', 'retired', 'blue-collar'])
        marital = st.selectbox("Marital Status", ['single', 'married', 'divorced'])
        education = st.selectbox("Education", ['primary', 'secondary', 'tertiary'])
        default = st.selectbox("Has Credit in Default?", ['no', 'yes'])
        balance = st.number_input("Balance", 0, 100000, 1000)

    with col2:
        housing = st.selectbox("Has Housing Loan?", ['no', 'yes'])
        loan = st.selectbox("Has Personal Loan?", ['no', 'yes'])
        contact = st.selectbox("Contact Type", ['cellular', 'telephone'])
        day = st.number_input("Last Contact Day", 1, 31, 15)
        month = st.selectbox("Month", ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])

    with col3:
        duration = st.number_input("Last Contact Duration (seconds)", 0, 5000, 200)
        campaign = st.number_input("Number of Contacts", 1, 50, 1)
        pdays = st.number_input("Days since last contact", -1, 1000, -1)
        previous = st.number_input("Previous Contacts", 0, 50, 0)
        poutcome = st.selectbox("Previous Outcome", ['unknown', 'failure', 'success'])

    if st.button("🚀 Predict Conversion"):
        input_data = pd.DataFrame([[age, job, marital, education, default, balance,
                                    housing, loan, contact, day, month, duration,
                                    campaign, pdays, previous, poutcome]], columns=columns)

        prob = model.predict_proba(input_data)[0][1]
        pred = (prob >= threshold)

        st.subheader("Result")
        st.write(f"Prediction: {'YES' if pred else 'NO'}")
        st.write(f"Probability: {prob:.2f}")

# ==========================
# Bulk Prediction
# ==========================
else:
    st.header("📂 Bulk Prediction Scanner")

    # Sample Download
    st.subheader("1. Download Sample Templates")

    sample_df = pd.DataFrame([
    {
        "age": 30,
        "job": "admin.",
        "marital": "married",
        "education": "secondary",
        "default": "no",
        "balance": 1000,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 15,
        "month": "may",
        "duration": 200,
        "campaign": 1,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown"
        }
    ])

    st.download_button("Download CSV Sample", sample_df.to_csv(index=False), "sample.csv")
    buffer = io.BytesIO()
    sample_df.to_excel(buffer, index=False, engine='openpyxl')

    st.download_button(
        label="Download Excel Sample",
        data=buffer.getvalue(),
        file_name="sample.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    st.download_button("Download JSON Sample", sample_df.to_json(orient="records"), "sample.json")

    # Upload
    st.subheader("2. Upload File")
    file = st.file_uploader("Upload CSV, Excel, or JSON", type=["csv", "xlsx", "json"])

    if file:
        if file.name.endswith("csv"):
            df = pd.read_csv(file)
        elif file.name.endswith("xlsx"):
            df = pd.read_excel(file)
        else:
            df = pd.read_json(file)

        st.write("Preview:")
        st.dataframe(df.head())

        # Validation
        missing_cols = [col for col in columns if col not in df.columns]

        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
        else:
            if st.button("Run Bulk Prediction"):
                probs = model.predict_proba(df)[:, 1]
                preds = (probs >= threshold)

                df['Prediction'] = np.where(preds, 'Yes', 'No')
                df['Probability'] = probs

                st.success("Prediction Complete")
                st.dataframe(df)

                csv = df.to_csv(index=False)
                st.download_button("Download Results", csv, "predictions.csv")
