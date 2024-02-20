import streamlit as st
import pandas as pd
import pickle

def predict_probabilities(data, model_path="model_xgb.sav"):
    try:
        with open(model_path, 'rb') as model_file:
            clf = pickle.load(model_file)
        probabilities = clf.predict_proba(data)[:, 1]  # Assuming the positive class is cancer
        return probabilities
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
            # Rearrange the columns and calculate the mean of 'plasma_CA19_9'
            processed_data = df[required_columns]
            processed_data['mean_CA19_9'] = processed_data['plasma_CA19_9'].mean()
            processed_data = processed_data[['REG1A', 'creatinine', 'TFF1', 'LYVE1', 'plasma_CA19_9', 'REG1B', 'age', 'mean_CA19_9']]

            # Get predicted probabilities using the pre-trained model
            probabilities = predict_probabilities(processed_data)
            
            st.subheader("Probability of Cancer for Each Row:")
            st.write(probabilities)

    else:
        st.warning("The uploaded CSV file does not have the expected column names for pancreatic cancer detection. Please check the file structure")
