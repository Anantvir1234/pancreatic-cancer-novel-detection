import streamlit as st
import pandas as pd
import pickle
import subprocess

# Check if xgboost is installed
try:
    import xgboost
except ImportError:
    st.error("xgboost not found. Attempting to install xgboost...")

    # Try installing xgboost
    try:
        subprocess.run(["pip", "install", "xgboost"])
        import xgboost  # Check the import again after installation
        st.success("xgboost has been successfully installed!")
    except Exception as install_error:
        st.error(f"Failed to install xgboost. Please install it manually with 'pip install xgboost' and then run the application. Error: {install_error}")
        st.stop()

def predict(data, model_path="model_xgb.sav"):
    try:
        with open(model_path, 'rb') as model_file:
            clf = pickle.load(model_file)
        predictions = clf.predict(data)
        return predictions
    except Exception as e:
        return f"Error: {e}"

def display_results(predictions):
    st.subheader("Final Results:")
    if any(predictions):
        st.write("Pancreatic Cancer Detected")
    else:
        st.write("Not Detected")

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
                # Get predictions using the pre-trained model
                predictions = predict(df[required_columns])
                display_results(predictions)
                st.write("Debug: Model Predictions:", predictions)
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

        # Get predictions using the pre-trained model
        predictions = predict(input_df[required_columns])
        display_results(predictions)
        st.write("Debug: Model Predictions:", predictions)
