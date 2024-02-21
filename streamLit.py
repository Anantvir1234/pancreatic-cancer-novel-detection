import streamlit as st
import pandas as pd
import pickle

def predict_proba(data, model_path="model_xgb.sav"):
    try:
        with open(model_path, 'rb') as model_file:
            clf = pickle.load(model_file)
            probabilities = clf.predict_proba(data)
        return probabilities
    except Exception as e:
        return f"Error: {e}"

# Title and description
title = "Pancreatic Cancer Detection"
st.set_page_config(page_title=title)
st.header(title)
st.markdown("Detect pancreatic cancer through a CSV file or input raw data")

session_state = st.session_state
if 'active_tab' not in session_state:
    session_state.active_tab = "Upload a .CSV"

if session_state.active_tab == "Upload a .CSV":
    # On the "Upload a .CSV" tab
    st.sidebar.header('Upload a CSV file')
    st.sidebar.markdown("Please upload a CSV file for detection.")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Preview of the uploaded data:")
        st.write(df.head())

        # Add required column names
        required_columns = ["REG1A", "creatinine", "TFF1", "LYVE1", "plasma_CA19_9", "REG1B", "age", "pancreatic_cancer"]

        if all(col in df.columns for col in required_columns):
            st.subheader("Pancreatic Cancer Detection Results:")
            if st.button("Process Uploaded File", disabled="error" in st.session_state):
                st.subheader("Final Results:")
                probabilities = predict_proba(df[required_columns])
                if isinstance(probabilities, str):
                    st.error(probabilities)
                else:
                    cancer_probabilities = probabilities[:, 1]
                    st.write("Pancreatic Cancer Probability:")
                    st.write(cancer_probabilities)
                    st.write("Pancreatic Cancer Detected" if any(cancer_probabilities > 0.5) else "Not Detected")
        else:
            st.warning("The uploaded CSV file does not have the expected column names for pancreatic cancer detection. Please check the file structure")

else:
    st.sidebar.header('Please Input Features Value')

    def user_input_features():
        age = st.sidebar.number_input('Age of persons: ', min_value=1)
        if age <= 0:
            st.error("Age should be greater than 0.")
            return None
        sex = st.sidebar.number_input('Gender of persons 1=Female, 2=Male: ', min_value=1, max_value=2, format="%d")
        if sex not in [1, 2]:
            st.error("Gender should be either 1 or 2.")
            return None
        ca_19_19 = st.sidebar.number_input('Plasma CA_19_9: ')
        creatinine = st.sidebar.number_input('Creatinine: ')
        LYVE1 = st.sidebar.number_input('LYVE1: ')
        REG1B = st.sidebar.number_input('REG1B: ')
        REG1A = st.sidebar.number_input('REG1A')
        TFF1 = st.sidebar.number_input('TFF1: ')
        data = {'age': age, 'sex': sex, 'plasma_CA19_9': ca_19_19, 'creatinine': creatinine, 'LYVE1': LYVE1,
                'REG1B': REG1B, 'REG1A': REG1A, 'TFF1': TFF1}
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()
    if input_df is not None:
        if st.button("Process values", disabled="error" in st.session_state):
            st.subheader("Final Results:")
            probabilities = predict_proba(input_df)
            if isinstance(probabilities, str):
                st.error(probabilities)
            else:
                cancer_probabilities = probabilities[:, 1]
                st.write("Pancreatic Cancer Probability:")
                st.write(cancer_probabilities)
                st.write("Pancreatic Cancer Detected" if any(cancer_probabilities > 0.5) else "Not Detected")
        st.write(input_df)

if st.button("Upload a .CSV"):
    session_state.active_tab = "Upload a .CSV"
if st.button("Input raw data"):
    session_state.active_tab = "Input Raw Data"
