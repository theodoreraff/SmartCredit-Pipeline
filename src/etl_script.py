import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os

# --- Scikit-learn Global Configuration ---
# Ensures transformers output Pandas DataFrames, maintaining column names.
from sklearn import set_config
set_config(transform_output="pandas")

# --- Database Configuration ---
DB_NAME = 'loan_prediction_db'
DB_USER = 'loan_pred_user'
DB_PASS = 'L0anP-red#2025!'
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_TABLE_NAME = 'loan_applications'

# --- Model & File Paths Configuration ---
# Dynamically determine paths relative to the project root for robustness.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_FILE_PATH = os.path.join(MODELS_DIR, 'loan_approval_model.pkl')

print("--- Starting Model Training Pipeline ---")

# --- 1. Fetch Clean Data from Database ---
print(f"1. Fetching clean data from '{DB_TABLE_NAME}' in '{DB_NAME}'...")
try:
    db_connection_str = f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    db_engine = create_engine(db_connection_str)
    loan_data_df = pd.read_sql_table(DB_TABLE_NAME, db_engine)
    print("   Data fetched successfully.")
except Exception as e:
    print(f"   ERROR: Failed to fetch data from PostgreSQL: {e}")
    print("   Ensure ETL script has been run and data exists in the database.")
    exit()

# --- 2. Prepare Data for Modeling ---
print("2. Preparing data for modeling...")
# Separate features (X) and target (y)
X = loan_data_df.drop('Loan_Status', axis=1)
y = loan_data_df['Loan_Status']

# Define categorical and numerical features for preprocessing
categorical_features = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']
numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'TotalIncome']

# Create preprocessing pipeline using ColumnTransformer
# OneHotEncoder for categoricals (sparse_output=False for Pandas compatibility)
# StandardScaler for numericals (handles naming and potential scaling)
data_preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ],
    remainder='passthrough', # Keep unlisted columns (though all should be covered)
    verbose_feature_names_out=False # Ensures clean output feature names
)
print("   Data preprocessing pipeline configured.")

# --- 3. Build and Train Model ---
print("3. Building and training the Machine Learning model...")
# Create a full pipeline: preprocessing + classifier
model_pipeline = Pipeline(steps=[('preprocessor', data_preprocessor),
                                 ('classifier', RandomForestClassifier(random_state=42))])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"   Data split: Training set {X_train.shape[0]} rows, Testing set {X_test.shape[0]} rows.")

# Train the model
model_pipeline.fit(X_train, y_train)
print("   Model training completed.")

# --- 4. Evaluate Model Performance ---
print("4. Evaluating model performance...")
y_pred = model_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"   Model Accuracy: {accuracy:.4f}")
print("\n   Classification Report:")
print(classification_report(y_test, y_pred))
print("   Model evaluation complete.")

# --- 5. Save Trained Model ---
print(f"5. Saving trained model to '{MODEL_FILE_PATH}'...")
try:
    os.makedirs(MODELS_DIR, exist_ok=True) # Ensure models directory exists
    with open(MODEL_FILE_PATH, 'wb') as file:
        pickle.dump(model_pipeline, file) # Save the entire pipeline
    print("   Model saved successfully.")
except Exception as e:
    print(f"   ERROR: Failed to save model: {e}")
    exit()

print("--- Model Training Pipeline Finished ---")