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
st.sidebar.header('Please Input Features Value')

# Collects user input features into dataframe

def user_input_features():
    age = st.sidebar.number_input('Age of persons: ')
    sex = st.sidebar.selectbox('Gender of persons 0=Female, 1=Male: ',(0,1))
    cp = st.sidebar.selectbox('Chest pain type (4 values)',(0,1,2,3))
    trtbps = st.sidebar.number_input('Resting blood pressure: ')
    chol = st.sidebar.number_input('Serum cholestrol in mg/dl: ')
    fbs =  st.sidebar.selectbox('Fasting blood sugar > 120 mg/dl:',( 0,1))
    restecg = st.sidebar.selectbox('Resting electrocardio results:', ( 0,1,2))
    thalachh = st.sidebar.number_input('Maximum heart rate achieved thalach: ')
    exng = st.sidebar.selectbox('Exercise induced angina: ',( 0,1))
    oldpeak = st.sidebar.number_input(' ST depression induced by exercise relative to rest (oldpeak): ')
    slp = st.sidebar.selectbox('The slope of the peak exercise ST segment (slp): ', ( 0,1,2))
    caa = st.sidebar.selectbox('Number of major vessels(0-3) colored by flourosopy (caa):',(0,1,2,3,4))
    thall = st.sidebar.selectbox(' Thall 0=normal, 1=fixed defect, 2 = reversable defect',(0,1,2,3))


    data = {'age':age, 'sex':sex, 'cp':cp, 'trtbps':trtbps, 'chol':chol, 'fbs':fbs, 'restecg':restecg, 'thalachh':thalachh,
       'exng':exng, 'oldpeak':oldpeak, 'slp':slp, 'caa':caa, 'thall':thall
                }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

st.write(input_df)

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
