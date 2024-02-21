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
st.markdown("Detect pancreatic cancer through an uploaded CSV file or input data on the left.")
upload_tab, input_tab = st.tabs(["Upload a .CSV", "Input raw data"])

with upload_tab:
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
        else:
            st.warning("The uploaded CSV file does not have the expected column names for pancreatic cancer detection. Please check the file structure")
with input_tab:
    st.sidebar.header('Please Input Features Value')
    # Collects user input features into dataframe
    def user_input_features():
        age = st.sidebar.number_input('Age of persons: ')
        sex = st.sidebar.selectbox('Gender of persons 0=Female, 1=Male: ',(0,1))
        ca_19_19 = st.sidebar.number_input('Plasma CA_19_9: ')
        creatinine = st.sidebar.number_input('Creatinine: ')
        LYVE1 = st.sidebar.number_input('LYVE1: ')
        REG1B =  st.sidebar.number_input('REG1B: ')
        REG1A = st.sidebar.number_input('REG1A')
        TFF1 = st.sidebar.number_input('TFF1: ')

        data = {'age':age, 'sex':sex, 'ca_19_19':ca_19_19, 'creatinine':creatinine, 'LYVE1':LYVE1, 'REG1B':REG1B, 'REG1A':REG1A, 'TFF1':TFF1,
                        }
        features = pd.DataFrame(data, index=[0])
        
        return features
        if st.button("Process values"):
                # Get predictions using the pre-trained model
                predictions = predict(features)
                st.subheader("Final Results:")
                st.write("Pancreatic Cancer Detected" if any(predictions) else "Not Detected")
    
    input_df = user_input_features()
    st.write(input_df)
