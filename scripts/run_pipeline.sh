#!/bin/bash

# --- Set Working Directory to Project Root ---
# This ensures all relative paths (src/, data/, models/) work correctly,
# regardless of where the script is executed from.
cd "$(dirname "$(dirname "$0")")"

# --- Miniconda Environment Configuration ---
CONDA_ENV_NAME="loan_pred_env"

# --- Python Script Locations (Relative to Project Root) ---
ETL_SCRIPT="src/etl_script.py"
TRAIN_MODEL_SCRIPT="src/train_model.py"
APP_SCRIPT="src/app.py"

echo "--- Starting Loan Prediction Project Pipeline ---"
echo "Ensuring Conda environment '${CONDA_ENV_NAME}' is available."
echo "------------------------------------------------"

# --- 1. Validate Conda Environment ---
echo "1. Validating Conda environment: ${CONDA_ENV_NAME}..."
# Check if 'conda' command exists
if ! command -v conda &> /dev/null; then
    echo "ERROR: Conda not found. Please ensure Miniconda/Anaconda is installed and in your PATH."
    exit 1
fi

# Check if the specific Conda environment exists
if ! conda env list | grep -q "${CONDA_ENV_NAME}"; then
    echo "ERROR: Conda environment '${CONDA_ENV_NAME}' not found."
    echo "Please create it with 'conda create -n ${CONDA_ENV_NAME} python=3.9' and install all necessary dependencies."
    exit 1
fi
echo "   Conda environment found."
echo "------------------------------------------------"

# --- 2. Run ETL Process ---
echo "2. Running ETL process (${ETL_SCRIPT})..."
# Use 'conda run' to execute the Python script within the correct Conda environment
conda run -n "${CONDA_ENV_NAME}" python "${ETL_SCRIPT}"

# Check the exit status of the last command. If not 0, an error occurred.
if [ $? -eq 0 ]; then
    echo "   ETL process completed successfully."
else
    echo "ERROR: ETL process failed. Check the logs above for details."
    exit 1
fi

echo "------------------------------------------------"

# --- 3. Run Model Training ---
echo "3. Running model training (${TRAIN_MODEL_SCRIPT})..."
# Use 'conda run' to execute the Python script within the correct Conda environment
conda run -n "${CONDA_ENV_NAME}" python "${TRAIN_MODEL_SCRIPT}"

if [ $? -eq 0 ]; then
    echo "   Model training completed. Model saved in 'models/' folder."
else
    echo "ERROR: Model training failed. Check the logs above for details."
    exit 1
fi

echo "------------------------------------------------"

# --- 4. Launch Streamlit Application ---
echo "4. Launching Streamlit application (${APP_SCRIPT})..."
echo "   The application will open in your browser (usually http://localhost:8501)."
echo "   Press Ctrl+C in this terminal to stop the Streamlit application."
# Use 'conda run' to execute the Streamlit app within the correct Conda environment.
# This command will block the terminal until manually stopped (Ctrl+C).
conda run -n "${CONDA_ENV_NAME}" streamlit run "${APP_SCRIPT}"

# Note: The lines below will only execute after the Streamlit application is stopped (Ctrl+C)
echo "------------------------------------------------"
echo "Streamlit application stopped."

echo "--- Loan Prediction Project Pipeline Finished ---"