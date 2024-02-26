import streamlit as st
import pandas as pd
import pickle

# Set page title
title = "Pancreatic Cancer Detection"
st.set_page_config(page_title=title)

def predict(data, model_path="model_xgb.sav"):
    try:
        with open(model_path, 'rb') as model_file:
            clf = pickle.load(model_file)
            predictions = clf.predict(data)
        return predictions
    except Exception as e:
        return f"Error: {e}"

# Title and description
st.image('image-removebg-preview (17).png')
st.header(title)
st.markdown("Detect pancreatic cancer through a CSV file or input raw data")

# Track active tab using simple Python variables
active_tab = st.sidebar.radio("Navigation", ["Upload a .CSV", "Input Raw Data"])

if active_tab == "Upload a .CSV":
    # On the "Upload a .CSV" tab
    st.sidebar.header('Upload a CSV file')
    st.sidebar.markdown("Please upload a CSV file for detection.")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Preview of the uploaded data:")
        st.write(df.head())
        required_columns = ["REG1A", "creatinine", "TFF1", "LYVE1", "plasma_CA19_9", "REG1B", "age"]

        if set(required_columns).issubset(df.columns):
            st.subheader("Pancreatic Cancer Detection Results:")
            if st.button("Process Uploaded File", disabled="error" in st.session_state):
                predictions = predict(df[required_columns])
                st.subheader("Final Results:")
                cancer_detected = any(predictions)
                
                if not isinstance(cancer_detected, str):
                    st.write("Pancreatic Cancer Detected" if cancer_detected else "Not Detected")
                    st.checkbox("Cancer Detected", value=cancer_detected, disabled=True)
                    st.checkbox("Cancer Not Detected", value=not cancer_detected, disabled=True)
                else:
                    st.error(cancer_detected)
        else:
            st.warning("The uploaded CSV file does not have all the required column names for pancreatic cancer detection. Please check the file structure")

else:
    # On the "Input Raw Data" tab
    st.sidebar.header('Please Input Features Value')
    
    def user_input_features():
        age = st.sidebar.number_input('Age of persons: ', min_value=1)
        if age <= 0:
            st.error("Age should be greater than 0.")
            return None
        sex = st.sidebar.number_input('Gender of persons 0=Female, 1=Male: ', min_value=0, max_value=1, format="%d")
        if sex not in [0, 1]:
            st.error("Gender should be either 0 or 1.")
            return None
        ca_19_19 = st.sidebar.number_input('Plasma CA_19_9: ')
        creatinine = st.sidebar.number_input('Creatinine: ')
        LYVE1 = st.sidebar.number_input('LYVE1: ')
        REG1B = st.sidebar.number_input('REG1B: ')
        REG1A = st.sidebar.number_input('REG1A')
        TFF1 = st.sidebar.number_input('TFF1: ')
        data = {'age': age, 'sex': sex, 'ca_19_19': ca_19_19, 'creatinine': creatinine, 'LYVE1': LYVE1,
                'REG1B': REG1B, 'REG1A': REG1A, 'TFF1': TFF1}
        features = pd.DataFrame(data, index=[0])
        return features
    
    input_df = user_input_features()
    if st.button("Process values", disabled="error" in st.session_state):
        if input_df is not None:
            predictions = predict(input_df)
            st.subheader("Final Results:")
            cancer_detected = bool(predictions[0])
            if not isinstance(cancer_detected, str):
                st.write("Pancreatic Cancer Detected" if cancer_detected else "Not Detected")
                st.checkbox("Cancer Detected", value=cancer_detected, disabled=True)
                st.checkbox("Cancer Not Detected", value=not cancer_detected, disabled=True)
        st.write(input_df)

# Add buttons to switch tabs
if st.button("Upload a .CSV"):
    active_tab = "Upload a .CSV"
if st.button("Input raw data"):
    active_tab = "Input Raw Data"
