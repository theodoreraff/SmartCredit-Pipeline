import streamlit as st
import pandas as pd
import pickle
import os

# --- Model & File Paths Configuration ---
# Get the directory of the current script (e.g., /home/theodores/PycharmProjects/loan_prediction/src)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level to the project root directory (e.g., /home/theodores/PycharmProjects/loan_prediction)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Define the full path to the 'models' folder
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Define the full path for the trained model file
MODEL_FILE_PATH = os.path.join(MODELS_DIR, 'loan_approval_model.pkl')

# --- Function to Load Model (cached for performance) ---
# @st.cache_resource: Streamlit caches the model after its first load,
# making the app faster on subsequent refreshes.
@st.cache_resource
def load_trained_model():
    """Loads the pre-trained Machine Learning model from a .pkl file."""
    try:
        with open(MODEL_FILE_PATH, 'rb') as file:
            model = pickle.load(file)
        st.success("Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error(f"ERROR: Model file '{MODEL_FILE_PATH}' not found.")
        st.info("Please ensure 'train_model.py' script has been run to train and save the model.")
        return None
    except Exception as e:
        st.error(f"ERROR loading model: {e}")
        return None


# Load the model when the application starts
model = load_trained_model()

# Stop the app execution if model loading fails
if model is None:
    st.stop()

# --- Streamlit Page Configuration ---
# Sets page title, browser tab icon, and layout.
st.set_page_config(
    page_title="Loan Eligibility Prediction",
    page_icon="💰",
    layout="centered",  # 'centered' (default) or 'wide'
    # initial_sidebar_state="auto" # Removed as sidebar is no longer used
)

# --- Custom CSS Styling ---
# Provides detailed control over the app's appearance (colors, fonts, spacing, buttons).
# Use with caution: 'unsafe_allow_html=True' allows raw HTML/CSS injection.
st.markdown(
    """
    <style>
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #4CAF50; /* Bright green */
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    h2 {
        color: #333333;
        font-size: 1.8em;
        border-bottom: 2px solid #EEEEEE;
        padding-bottom: 0.3em;
        margin-top: 1.5em;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 1.1em;
        border: none;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.3);
    }
    /* Styling for text input/select boxes */
    .stTextInput>div>div>input, .stSelectbox>div>div>select {
        border-radius: 5px;
        border: 1px solid #DDDDDD;
        padding: 0.5em;
    }
    /* Styling for Streamlit status messages */
    .stSuccess {
        background-color: #e6ffe6;
        color: #3c763d;
        border-left: 5px solid #4CAF50;
        padding: 10px;
        border-radius: 5px;
    }
    .stError {
        background-color: #ffe6e6;
        color: #a94442;
        border-left: 5px solid #a94442;
        padding: 10px;
        border-radius: 5px;
    }
    .stInfo {
        background-color: #e6f7ff;
        color: #31708f;
        border-left: 5px solid #2196F3;
        padding: 10px;
        border-radius: 5px;
    }
    /* Adjustments for Streamlit's dark theme */
    [data-theme="dark"] .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    [data-theme="dark"] .stButton>button:hover {
        background-color: #45a049;
    }
    [data-theme="dark"] h1, [data-theme="dark"] h2, [data-theme="dark"] p, [data-theme="dark"] div, [data-theme="dark"] span, [data-theme="dark"] label {
        color: #F0F2F6; /* Lighter text color for dark theme */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Application Header ---
# Optional: Add your logo here if available (uncomment and adjust LOGO_PATH)
# if os.path.exists(LOGO_PATH):
#     st.image(LOGO_PATH, width=150)
st.markdown("<h1>💰 Loan Eligibility Prediction 💰</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 1.1em; color: #555555;'>This application predicts the eligibility of loan applicants based on historical data.</p>",
    unsafe_allow_html=True)

st.write("---")  # Simple horizontal line separator

# --- User Input Form ---
st.header("Enter Applicant Details")

# Use st.form to group inputs and a submit button.
# This ensures prediction only runs after explicit submission, not on every input change.
with st.form("loan_application_form"):
    # Layout using columns for a cleaner, minimalist appearance.
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Personal & Demographic Information")
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Marital Status", ["No", "Yes"])
        # Dependents value will be processed to integer before model prediction
        dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])

    with col2:
        st.subheader("Financial & Other Information")
        applicant_income = st.number_input("Applicant Income (IDR)", min_value=0, value=5000000, step=1000000)
        coapplicant_income = st.number_input("Co-Applicant Income (IDR)", min_value=0, value=0, step=1000000)
        loan_amount = st.number_input("Loan Amount Requested (IDR)", min_value=100000, value=15000000, step=1000000)
        loan_amount_term = st.selectbox("Loan Term (Months)", [12, 36, 60, 120, 180, 240, 300, 360, 480])
        credit_history = st.selectbox("Credit History (1.0 = Good, 0.0 = Bad)", [1.0, 0.0],
                                      format_func=lambda x: "Good" if x == 1.0 else "Bad")
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    # Form submission button
    submitted = st.form_submit_button("Check Eligibility")

    # --- Prediction Logic (triggered on form submission) ---
    if submitted:
        # Display a spinner while prediction is in progress
        with st.spinner('Processing data and making prediction...'):
            try:
                # Calculate 'TotalIncome' as it's a new feature used by the model
                total_income = float(applicant_income + coapplicant_income)

                # --- IMPORTANT: Preprocess 'Dependents' to integer, matching ETL ---
                processed_dependents = int(dependents.replace('3+', '3'))
                # -------------------------------------------------------------------

                # Prepare input data as a dictionary
                input_data_dict = {
                    'Gender': gender,
                    'Married': married,
                    'Dependents': processed_dependents,  # Use the processed integer value
                    'Education': education,
                    'Self_Employed': self_employed,
                    'ApplicantIncome': float(applicant_income),
                    'CoapplicantIncome': float(coapplicant_income),
                    'LoanAmount': float(loan_amount),
                    'Loan_Amount_Term': float(loan_amount_term),
                    'Credit_History': float(credit_history),
                    'Property_Area': property_area,
                    'TotalIncome': total_income
                }

                # Create a DataFrame from the dictionary.
                # CRITICAL: Ensure the column order matches the exact order of features (X)
                # used during model training in train_model.py.
                expected_columns_order = [
                    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                    'Credit_History', 'Property_Area', 'TotalIncome'
                ]

                # Create DataFrame and enforce the exact column order
                input_df = pd.DataFrame([input_data_dict])[expected_columns_order]

                # Make prediction using the loaded model pipeline
                prediction_numeric = model.predict(input_df)[0]
                # Get probability for the '1' class (Eligible)
                prediction_proba = model.predict_proba(input_df)[:, 1][0]

                st.write("---")  # Separator line

                # Display prediction results with distinct styling
                if prediction_numeric == 1:
                    st.success(f"🎉 Congratulations! Applicant is **Eligible for Loan!**")
                    st.info(f"Approval Probability: **{prediction_proba * 100:.2f}%**")
                    st.balloons()  # Fun balloon effect for approval
                else:
                    st.error(f"😞 Sorry, Applicant is **Not Eligible for Loan.**")
                    st.info(f"Approval Probability: **{prediction_proba * 100:.2f}%**")

                st.markdown("---")
                st.subheader("Input Details Used:")
                st.dataframe(input_df)  # Display the user's input data in a table

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.warning("Please check your input and try again.")