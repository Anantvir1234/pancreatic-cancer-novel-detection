import streamlit as st
import pandas as pd
import pickle

def predict(data, model_path="model_xgb.sav"):
    try:
        with open(model_path, 'rb') as model_file:
            clf = pickle.load(model_file)
            predictions = clf.predict(data)
            probabilities = clf.predict_proba(data)[:, 1]  # Probabilities of positive class (cancer detected)
        return predictions, probabilities
    except Exception as e:
        return None, f"Error: {e}"

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
        required_columns = ["REG1A", "creatinine", "TFF1", "LYVE1", "plasma_CA19_9", "REG1B", "age"]
        if all(col in df.columns for col in required_columns):
            st.subheader("Pancreatic Cancer Detection Results:")
            if st.button("Process Uploaded File"):
                predictions, probabilities = predict(df[required_columns])
                if predictions is not None:
                    st.subheader("Final Results:")
                    cancer_detected = any(predictions)
                    st.write("Pancreatic Cancer Detected" if cancer_detected else "Not Detected")
                    st.checkbox("Cancer Detected", value=cancer_detected, disabled=True)
                    st.checkbox("Cancer Not Detected", value=not cancer_detected, disabled=True)
                    st.subheader("Accuracy:")
                    accuracy = sum(predictions) / len(predictions)
                    st.write(f"Accuracy: {accuracy * 100:.2f}%")
        else:
            st.warning("The uploaded CSV file does not have the expected column names for pancreatic cancer detection. Please check the file structure")

else:
    st.sidebar.header('Please Input Features Value')
    
    def user_input_features():
        age = st.sidebar.number_input('Age of persons: ')
        sex = st.sidebar.selectbox('Gender of persons 0=Female, 1=Male: ', (0, 1))
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
    if st.button("Process values"):
        predictions, probabilities = predict(input_df)
        if predictions is not None:
            st.subheader("Final Results:")
            cancer_detected = bool(predictions[0])
            st.write("Pancreatic Cancer Detected" if cancer_detected else "Not Detected")
            st.checkbox("Cancer Detected", value=cancer_detected, disabled=True)
            st.checkbox("Cancer Not Detected", value=not cancer_detected, disabled=True)
            st.subheader("Accuracy:")
            accuracy = probabilities[0] if cancer_detected else 1 - probabilities[0]
            st.write(f"Accuracy: {accuracy * 100:.2f}%")
    st.write(input_df)

if st.button("Upload a .CSV"):
    session_state.active_tab = "Upload a .CSV"
if st.button("Input raw data"):
    session_state.active_tab = "Input Raw Data"



