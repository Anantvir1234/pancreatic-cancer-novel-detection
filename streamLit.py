import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

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
                
                # Step 2: Threshold selection
                threshold = 50  # Example threshold value (you need to choose based on your data)
                
                # Step 3: Visualization
                plt.hist(df[df['pancreatic_cancer'] == 1]['plasma_CA19_9'], bins=20, color='red', alpha=0.5, label='Pancreatic Cancer Detected')
                plt.hist(df[df['pancreatic_cancer'] == 0]['plasma_CA19_9'], bins=20, color='blue', alpha=0.5, label='No Pancreatic Cancer Detected')
                plt.axvline(x=threshold, color='black', linestyle='--', label=f'Threshold = {threshold}')
                plt.xlabel('CA19-9 Levels')
                plt.ylabel('Frequency')
                plt.title('Distribution of CA19-9 Levels for Pancreatic Cancer Detection')
                plt.legend()
                st.pyplot()
                
                # Step 4: Detection logic
                df['cancer_detected'] = df['plasma_CA19_9'] > threshold

                st.subheader("Final Results:")
                st.write(df)  # Output DataFrame with added cancer_detected column
                
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
            
            # Step 2: Threshold selection
            threshold = 50  # Example threshold value (you need to choose based on your data)

            st.subheader("Final Results:")
            cancer_detected = bool(input_df['plasma_CA19_9'][0] > threshold)  # Assuming positive class is index 1
            if not isinstance(cancer_detected, str):
                st.write("Pancreatic Cancer Detected" if cancer_detected else "Not Detected")
                st.checkbox("Cancer Detected", value=cancer_detected, disabled=True)
                st.checkbox("Cancer Not Detected", value=not cancer_detected, disabled=True)
        st.write(input_df)

if st.button("Upload a .CSV"):
    session_state.active_tab = "Upload a .CSV"
if st.button("Input raw data"):
    session_state.active_tab = "Input Raw Data"

