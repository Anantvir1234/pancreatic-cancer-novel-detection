import streamlit as st
import pandas as pd
import joblib
import sys
import subprocess

# Install lightgbm if not already installed
try:
    import lightgbm
except ImportError:
    st.warning("Installing LightGBM...")
    subprocess.run([sys.executable, "-m", "pip", "install", "lightgbm"])
    st.success("LightGBM installed successfully!")

# Set page configuration
title = "Pancreatic Cancer Detection"
st.set_page_config(page_title=title)



# Define clf at the beginning of the script
clf = None

def load_model(model_path="best_model_lgbm.sav"):
    global clf  # Declare clf as a global variable
    try:
        clf = joblib.load(model_path)
        return clf
    except Exception as e:
        return f"Error loading model: {e}"

def predict(data):
    global clf  # Declare clf as a global variable
    try:
        predictions = clf.predict(data)
        return predictions
    except Exception as e:
        return f"Error making predictions: {e}"

# Title and description
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
        required_columns = ["REG1A", "creatinine", "TFF1", "LYVE1", "plasma_CA19_9", "REG1B", "age", "sex"]
        if all(col in df.columns for col in required_columns):
            st.subheader("Pancreatic Cancer Detection Results:")

            # Load model if not loaded
            if clf is None:
                clf = load_model()

            # Button for processing the uploaded file
            if st.button("Process Uploaded File", key="process_uploaded_file"):
                # Get probabilities of positive class using the pre-trained model
                predictions_proba = predict(df[required_columns])
                threshold = 0.3  # You can adjust this threshold based on your model and requirements

                # Convert numpy array to Pandas Series
                predictions_proba_series = pd.Series(predictions_proba)

                # Convert the elements to float and fill NaN values with 0
                predictions_proba_numeric = pd.to_numeric(predictions_proba_series, errors='coerce').fillna(0)

                # Display model information
                st.subheader("Loaded Model Information:")
                st.write(clf)  # Display model information

                # Convert probabilities to binary predictions using the threshold
                predictions = (predictions_proba_numeric.astype(float) > threshold).astype(int)

                st.subheader("Final Results:")
                st.write("Pancreatic Cancer Detected" if any(predictions) else "Not Detected")

        else:
            st.warning("The uploaded CSV file does not have the expected column names for pancreatic cancer detection. Please check the file structure")

else:
    # Input raw data
    st.subheader("Please Input Features Value")

    # Input numerical values for each column and biomarker
    features_input = {}
    required_columns = ["REG1A", "creatinine", "TFF1", "LYVE1", "plasma_CA19_9", "REG1B", "age", "sex"]

    for column in required_columns:
        if column == "age":
            features_input[column] = st.number_input(f'{column} (greater than or equal to 1): ', min_value=1)
        elif column == "sex":
            features_input[column] = st.number_input(f'{column} (0 for Male, 1 for Female): ', min_value=0, max_value=1, format="%d")
        else:
            features_input[column] = st.number_input(f'{column}: ', min_value=0)

    # Button for processing the inputted raw data
    if st.button("Process Raw Data", key="process_raw_data"):
        # Create a DataFrame with the input data
        input_df = pd.DataFrame(features_input, index=[0])

        # Get predictions using the pre-trained model
        predictions = predict(input_df[required_columns])
        st.subheader("Final Results:")
        st.write("Pancreatic Cancer Detected" if any(predictions) else "Not Detected")
