import streamlit as st
import pandas as pd
import joblib

# === Page Config ===
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

st.title("Customer Churn Prediction App")
st.write("This app predicts whether a **customer** is likely to **Churn** or remain **Loyal** based on their Profile and Financial Information.")

# === Load Model ===
model = joblib.load("model_cust_churn.joblib")

def get_prediction(data: pd.DataFrame):
    pred = model.predict(data)
    pred_proba = model.predict_proba(data)
    return pred, pred_proba

# === Layout: 3 Columns ===
col1, col2, col3 = st.columns(3, gap="large")

# --- Customer Profile ---
with col1:
    st.subheader("👤 Customer Profile")
    surname = st.text_input("Surname (optional)", "")
    age = st.slider("Age", min_value=18, max_value=92, value=30)
    gender = st.radio("Gender", ["Male", "Female"])
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

# --- Financial Features ---
with col2:
    st.subheader("💰 Financial Information")
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650, step=1)
    balance = st.number_input("Balance (USD)", min_value=0.0, value=50000.0, step=1.0)
    estimated_salary = st.number_input("Estimated Salary (USD)", min_value=0.0, value=60000.0, step=1.0)

# --- Business / Engagement ---
with col3:
    st.subheader("📈 Business & Engagement")
    tenure = st.slider("Tenure (Years with Bank)", min_value=0, max_value=10, value=5)
    num_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=2)
    has_cr_card = st.radio("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active_member = st.radio("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

# === Data Preparation ===
input_data = {
    "Surname": [surname.strip() if surname.strip() else "Not Provided"],
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
}

data = pd.DataFrame(input_data)

# === Convert binary columns to Yes/No for display ===
data_display = data.copy()
data_display["HasCrCard"] = data_display["HasCrCard"].map({1: "Yes", 0: "No"})
data_display["IsActiveMember"] = data_display["IsActiveMember"].map({1: "Yes", 0: "No"})

# === Rename columns for display ===
display_columns = {
    "Surname": "Customer Surname",
    "CreditScore": "Credit Score",
    "Geography": "Geography",
    "Gender": "Gender",
    "Age": "Age (Years)",
    "Tenure": "Bank Tenure (Years)",
    "Balance": "Account Balance (USD)",
    "NumOfProducts": "Number of Products",
    "HasCrCard": "Credit Card",
    "IsActiveMember": "Active Member",
    "EstimatedSalary": "Estimated Annual Salary (USD)"
}
data_display = data_display.rename(columns=display_columns)


# === Custom Formatters ===
formatters = {
    "Credit Score": lambda x: f"{x:,.0f}",
    "Age (Years)": lambda x: f"{x:,.0f}",
    "Bank Tenure (Years)": lambda x: f"{x:,.0f}",
    "Account Balance (USD)": lambda x: f"{x:,.2f}",
    "Number of Products": lambda x: f"{x:,.0f}",
    "Estimated Annual Salary (USD)": lambda x: f"{x:,.2f}"}

# === Show Input Data (Centered + Clean) ===
st.write("### Input Data Preview")

styled = (
    data_display.style
    .format(formatters)
    .set_table_styles(
        [
            {"selector": "th", "props": [("text-align", "center !important")]},  # Header rata tengah
            {"selector": "td", "props": [("text-align", "center !important")]}   # Nilai rata tengah
        ]
    )
)

st.markdown(
    styled.to_html(index=False).replace(
        '<table border="1" class="dataframe">',
        '<table style="margin-left:auto;margin-right:auto;text-align:center;border-collapse:collapse;" border="1" class="dataframe">'
    ),
    unsafe_allow_html=True
)



# === Prediction Button (Styled with CSS) ===
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)
if st.button("Predict Customer Churn"):
    # Drop Surname from prediction
    data_for_pred = data.drop(columns=["Surname"])

    # Predict
    pred, pred_proba = get_prediction(data_for_pred)
    label_map = {0: "Loyal", 1: "Churn"}

    label_pred = label_map[pred[0]]
    proba_loyal = pred_proba[0][0]
    proba_churn = pred_proba[0][1]

    # Show result
    st.subheader("Prediction Result")
    st.write(f"**Customer Name**: {data['Surname'][0]}")

    if pred[0] == 1:
        st.error("The customer is likely to **CHURN**")
    else:
        st.success("The customer is likely to remain **LOYAL**")

    # Show probabilities
    st.write("### Prediction Probabilities")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Loyal Probability", value=f"{proba_loyal:.0%}")
        st.progress(int(proba_loyal * 100))

    with col2:
        st.metric(label="Churn Probability", value=f"{proba_churn:.0%}")
        st.progress(int(proba_churn * 100))

