import streamlit as st
import pandas as pd
import pickle


def predict(data, model_path="model_xgb.sav"):
    try:
        with open(model_path, 'rb') as model_file:
            clf = pickle.load(model_file)
        predictions = clf.predict(data)
        return predictions
    except Exception as e:
        return f"Error: {e}"

# Title and description
title = "Pancreatic Cancer Detection"
st.set_page_config(page_title=title)
st.header(title)
st.markdown("Detect pancreatic cancer through an uploaded CSV file.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    # Load CSV data into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Display the first few rows of the DataFrame
    st.subheader("Preview of the uploaded data:")
    st.write(df.head())

    # Check for specific column names relevant to pancreatic cancer detection
    required_columns = ["REG1A", "creatinine", "TFF1", "LYVE1", "plasma_CA19_9", "REG1B", "age"]
    if all(col in df.columns for col in required_columns):
        st.subheader("Pancreatic Cancer Detection Results:")

        # Button for processing the uploaded file
        if st.button("Process Uploaded File"):
            # Get predictions using the pre-trained model
            predictions = predict(df[required_columns])
            st.subheader("Final Results:")
            st.write("Pancreatic Cancer Detected" if any(predictions) else "Not Detected")
            # In the predict_pancreatic_cancer function:

           def predict_pancreatic_cancer(data, model_path="model_xgb.sav"):
           # Load the pre-trained model using pickle
           with open(model_path, 'rb') as model_file:
           clf = pickle.load(model_file)
           features = ["REG1A", "creatinine", "TFF1", "LYVE1", "plasma_CA19_9", "REG1B", "age"]
           input_data = data[features]

           # Make probability predictions
           prediction_probs = clf.predict_proba(input_data)

          return prediction_probs
          # In the processing section:
          prediction_probs = predict_pancreatic_cancer(df)
          st.subheader("Model Prediction Probabilities:")
          st.write(prediction_probs[:, 1])  # Assuming 1 is the cancer class, adjust if needed
    else:
        st.warning("The uploaded CSV file does not have the expected column names for pancreatic cancer detection. Please check the file structure")
