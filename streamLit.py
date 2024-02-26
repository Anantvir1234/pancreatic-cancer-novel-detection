import streamlit as st
import pandas as pd
import xgboost as xgb
import subprocess
import sys

def train_model():
    # Replace these placeholders with your actual training data and labels
    train_data = pd.DataFrame({
        "REG1A": [1911.565138],
        "creatinine": [1.31196],
        "TFF1": [369.344],
        "LYVE1": [5.917939],
        "plasma_CA19_9": [1916],
        "REG1B": [381.221725],
        "age": [73],
        "gender": [0]  # Assuming M=0, F=1 for gender
    })
    labels = pd.Series([3])  # Assuming the diagnosis label is 3 based on the provided dataset

    clf = xgb.XGBClassifier()
    clf.fit(train_data, labels)

    # Save the model
    clf.save_model("model_xgb.json")

# Check if xgboost is installed, if not, attempt to install
try:
    import xgboost
except ImportError:
    st.error("xgboost not found. Attempting to install xgboost...")

    # Install xgboost
    try:
        st.warning("Installing xgboost. This may take a few moments...")
        subprocess.run([sys.executable, "-m", "pip", "install", "xgboost"])
        import xgboost  # Check the import again after installation
        st.success("xgboost has been successfully installed!")
    except Exception as install_error:
        st.error(f"Failed to install xgboost. Please install it manually with 'pip install xgboost' and then run the application. Error: {install_error}")
        st.stop()

# Run training function only when the app is loaded for the first time
if 'model_trained' not in st.session_state:
    train_model()
    st.session_state.model_trained = True

# Load the model outside the Streamlit app to avoid retraining on every run
model = xgb.XGBClassifier()
model.load_model("model_xgb.json")

# Title and description
title = "Pancreatic Cancer Detection"
st.set_page_config(page_title=title)
st.header(title)
st.markdown("Detect pancreatic cancer through an uploaded CSV file or input raw data.")

# Choose between uploading a CSV file or inputting raw data
option = st.radio("Select an option:", ["Upload a CSV file", "Input Raw Data"])

if option == "Upload a CSV file":
    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        # Load CSV data into a DataFrame
        df = pd.read_csv(uploaded_file)
        st.subheader("Preview of the uploaded data:")
        st.write(df.head().values.tolist())

        # Check for specific column names relevant to pancreatic cancer detection
        required_columns = ["REG1A", "creatinine", "TFF1", "LYVE1", "plasma_CA19_9", "REG1B", "age", "gender"]
        if all(col in df.columns for col in required_columns):
            st.subheader("Pancreatic Cancer Detection Results:")

            # Button for processing the uploaded file
            if st.button("Process Uploaded File", key="process_uploaded_file"):
                # Get probabilities of positive class using the pre-trained model
                predictions_proba = model.predict_proba(df[required_columns])[:, 1]
                threshold = 0.5  # Set the threshold for detection to 50%

                # Convert probabilities to binary predictions using the threshold
                predictions = (predictions_proba > threshold).astype(int)

                st.subheader("Final Results:")
                st.write("Pancreatic Cancer Detected" if any(predictions) else "Not Detected")
        else:
            st.warning("The uploaded CSV file does not have the expected column names for pancreatic cancer detection. Please check the file structure")

else:
    # Input raw data
    st.subheader("Please Input Features Value")

    # Input numerical values for each column and biomarker
    features_input = {}
    required_columns = ["REG1A", "creatinine", "TFF1", "LYVE1", "plasma_CA19_9", "REG1B", "age", "gender"]

    for column in required_columns:
        if column == "age":
            features_input[column] = st.number_input(f'{column} (greater than or equal to 1): ', min_value=1)
        elif column == "gender":
            features_input[column] = st.number_input(f'{column} (0 for Male, 1 for Female): ', min_value=0, max_value=1, format="%d")
        else:
            features_input[column] = st.number_input(f'{column}: ', min_value=0)

    # Button for processing the inputted raw data
    if st.button("Process Raw Data", key="process_raw_data"):
        # Create a DataFrame with the input data
        input_df = pd.DataFrame(features_input, index=[0])

        # Get probabilities of positive class using the pre-trained model
        predictions_proba = model.predict_proba(input_df[required_columns])[:, 1]
        threshold = 0.5  # Set the threshold for detection to 50%

        # Convert probabilities to binary predictions using the threshold
        predictions = (predictions_proba > threshold).astype(int)

        st.subheader("Final Results:")
        st.write("Pancreatic Cancer Detected" if any(predictions) else "Not Detected")
