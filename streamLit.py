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
            if st.button("Process Uploaded File", key="uploaded_file_button"):
                # Get predictions using the pre-trained model
                predictions = predict(df[required_columns])
                st.subheader("Final Results:")
                st.write("Pancreatic Cancer Detected" if any(predictions) else "Not Detected")
        else:
            st.warning("The uploaded CSV file does not have the expected column names for pancreatic cancer detection. Please check the file structure")

else:
    # Input raw data
    st.subheader("Please Input Features Value")

    # Input numerical values for each column
    biomarkers = ["REG1A", "creatinine", "TFF1", "LYVE1", "plasma_CA19_9", "REG1B"]
    input_values = {}
    for biomarker in biomarkers:
        input_values[biomarker] = st.number_input(f"{biomarker}: ", min_value=0)

    # Input for age and gender
    age = st.number_input('Age of persons: ', min_value=1)
    gender = st.number_input('Gender of persons (0=Male, 1=Female): ', min_value=0, max_value=1, format="%d")

    if age < 1:
        st.error("Age should be greater than or equal to 1.")
    elif gender not in [0, 1]:
        st.error("Gender should be either 0 or 1.")
    else:
        input_values["age"] = age
        input_values["gender"] = gender

        # Button for processing the inputted raw data
        if st.button("Process Raw Data", key="raw_data_button"):
            # Create a DataFrame with the input data
            input_df = pd.DataFrame(input_values, index=[0])

            # Get predictions using the pre-trained model
            predictions = predict(input_df[required_columns])
            st.subheader("Final Results:")
            st.write("Pancreatic Cancer Detected" if any(predictions) else "Not Detected")



