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
DB_TABLE_NAME = 'loan_applications' # Name of the table where cleaned data is stored

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
    # Create connection string for SQLAlchemy
    db_connection_str = f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    db_engine = create_engine(db_connection_str)

    # Read data from the 'loan_applications' table into a Pandas DataFrame
    loan_data_df = pd.read_sql_table(DB_TABLE_NAME, db_engine)
    print("   Data fetched successfully.")
    # Removed df.head() and df.info() for cleaner final output, but useful for debugging.

except Exception as e:
    print(f"   ERROR: Failed to fetch data from PostgreSQL: {e}")
    print("   Ensure ETL script has been run and data exists in the database.")
    exit()

print("\n" + "-" * 30)

# --- 2. Prepare Data for Modeling ---
print("2. Preparing data for modeling...")

# Separate features (X) and target (y)
# 'Loan_Status' is our target column
X = loan_data_df.drop('Loan_Status', axis=1)
y = loan_data_df['Loan_Status']
print(f"   Features (X) have {X.shape[1]} columns and {X.shape[0]} rows.")
print(f"   Target (y) has {y.shape[0]} rows.")

# Identify categorical and numerical columns for appropriate preprocessing.
# 'Dependents' is treated as categorical for OneHotEncoder, despite being integer.
categorical_features = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']
numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'TotalIncome']

# Create preprocessing pipeline using ColumnTransformer.
# This is a standard way to apply different transformations to different columns.
data_preprocessor = ColumnTransformer(
    transformers=[
        # OneHotEncoder for categorical features.
        # 'sparse_output=False' is crucial for compatibility with 'transform_output="pandas"'.
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        # StandardScaler for numerical features.
        # This handles column naming consistency when 'transform_output="pandas"' is set.
        ('num', StandardScaler(), numerical_features)
    ],
    remainder='passthrough', # Ensures columns not explicitly listed are passed through (though all should be covered).
    verbose_feature_names_out=False # Ensures cleaner output feature names from the transformer.
)

print("   Data preprocessing pipeline configured.")

print("\n" + "-" * 30)

# --- 3. Build and Train Model ---
print("3. Building and training the Machine Learning model...")

# Create a complete pipeline that combines preprocessing and the classifier.
# This ensures the same data transformations are applied consistently to new data.
model_pipeline = Pipeline(steps=[('preprocessor', data_preprocessor),
                                 ('classifier', RandomForestClassifier(random_state=42))]) # Random Forest Classifier chosen for this project.

# Split data into training and testing sets.
# 'test_size=0.2' means 20% of the data will be used for testing.
# 'random_state=42' ensures reproducible results.
# 'stratify=y' ensures the proportion of target classes (Y/N) is balanced in both sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"   Data split: Training set {X_train.shape[0]} rows, Testing set {X_test.shape[0]} rows.")

# Train the model using the training data.
model_pipeline.fit(X_train, y_train)
print("   Model training completed.")

print("\n" + "-" * 30)

# --- 4. Evaluate Model Performance ---
print("4. Evaluating model performance...")

# Make predictions on the testing set.
y_pred = model_pipeline.predict(X_test)

# Calculate and display the model's accuracy.
accuracy = accuracy_score(y_test, y_pred)
print(f"   Model Accuracy: {accuracy:.4f}")

# Display a comprehensive classification report.
# This provides precision, recall, and f1-score metrics for each class.
print("\n   Classification Report:")
print(classification_report(y_test, y_pred))
print("   Model evaluation complete.")

print("\n" + "-" * 30)

# --- 5. Save Trained Model ---
print(f"5. Saving trained model to '{MODEL_FILE_PATH}'...")
try:
    # Create the 'models' folder if it doesn't exist (using the absolute path).
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save the entire model_pipeline object using pickle.
    # 'wb' means 'write binary'.
    with open(MODEL_FILE_PATH, 'wb') as file:
        pickle.dump(model_pipeline, file)
    print("   Model saved successfully.")

except Exception as e:
    print(f"   ERROR: Failed to save model: {e}")
    exit()

print("\n--- Model Training Pipeline Finished ---")