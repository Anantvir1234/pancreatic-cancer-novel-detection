import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb

def predict(data, model_path="model_xgb.sav"):
    with open(model_path, 'rb') as model_file:
        clf = pickle.load(model_file)
    return clf.predict(data)

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

            # Assuming you have ground truth labels in a column named "ground_truth" in your DataFrame
            ground_truth_labels = df["ground_truth"]

            # Evaluate accuracy
            accuracy = sum(predictions == ground_truth_labels) / len(ground_truth_labels)

            # Display accuracy
            st.subheader(f"Model Accuracy: {accuracy * 100:.2f}%")

    else:
        st.warning("The uploaded CSV file does not have the expected column names for pancreatic cancer detection. Please check the file structure")
