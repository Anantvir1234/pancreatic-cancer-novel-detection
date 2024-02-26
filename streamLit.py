import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb

# Define clf at the beginning of the script
clf = None

# Custom session state class
class SessionState:
    def __init__(self, **kwargs):
        self._state = kwargs

    def __getattr__(self, attr):
        return self._state.get(attr, None)

    def __setattr__(self, attr, value):
        self._state[attr] = value

# Create a custom session state
state = SessionState(model_trained=False)

# Function to train the model
def train_model():
    # Replace these placeholders with your actual training data and labels
    train_data = pd.DataFrame({
        "REG1A": [1911.565138],
        "creatinine": [1.31196],
        "TFF1": [369.344],
        "LYVE1": [5.917939],
        "plasma_CA19_9": [1916],
        "REG1B": [381.221725],
        "age": [73],
        "gender": [0]  # Assuming M=0, F=1 for gender
    })
    labels = pd.Series([0])  # Assuming the diagnosis label is 0 for no cancer based on the provided dataset

    clf = xgb.XGBClassifier()
    clf.fit(train_data, labels)

    # Save the model using Booster.save_model
    clf.get_booster().save_model("model_xgb.json")

# Function to load the model
def load_model(model_path="model_xgb.json"):
    try:
        booster = xgb.Booster()
        booster.load_model(model_path)
        return booster
    except Exception as e:
        return f"Error loading model: {e}"

# Function to make predictions
def predict(data, model):
    try:
        predictions_proba = model.predict_proba(data)[:, 1]  # Probabilities of positive class
        return predictions_proba
    except Exception as e:
        return f"Error making predictions: {e}"

# Title and description
title = "Pancreatic Cancer Detection"
st.set_page_config(page_title=title)
st.header(title)
st.markdown("Detect pancreatic cancer through an uploaded CSV file or input raw data.")

# Run training function only when the app is loaded for the first time
if not state.model_trained:
    train_model()
    state.model_trained = True

# Load the model outside the Streamlit app to avoid retraining on every run
model = load_model()

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

        # Print column names for debugging
        st.write("Columns in the uploaded file:", df.columns.tolist())

        # Extract common columns with required_columns
        common_columns = list(set(required_columns) & set(df.columns))

        if all(col in df.columns for col in common_columns):
            st.subheader("Pancreatic Cancer Detection Results:")

            # Button for processing the uploaded file
            if st.button("Process Uploaded File", key="process_uploaded_file"):
                # Get probabilities of positive class using the pre-trained model
                predictions_proba = predict(df[common_columns], model)
                threshold = 0.5  # Set the threshold for detection to 50%

                # Convert numpy array to Pandas Series
                predictions_proba_series = pd.Series(predictions_proba)

                # Convert the elements to float and fill NaN values with 0
                predictions_proba_numeric = pd.to_numeric(predictions_proba_series, errors='coerce').fillna(0)

                # Convert probabilities to binary predictions using the threshold
                predictions = (predictions_proba_numeric.astype(float) > threshold).astype(int)

                st.subheader("Final Results:")
                st.write("Pancreatic Cancer Detected" if any(predictions) else "Not Detected")

            # Display model information
            st.subheader("Loaded Model Information:")
            st.write(clf)

        else:
            st.warning("The uploaded CSV file does not have the expected common columns for pancreatic cancer detection. Please check the file structure and make sure the necessary columns are present.")

else:
    # Input raw data
    st.subheader("Please Input Features Value")

    # Input numerical values for each column and biomarker
    features_input = {}
    for column in required_columns:
        if column == "age":
            features_input[column] = st.number_input(f'{column} (greater than or equal to 1): ', min_value=1)
        elif column == "gender":
            features_input[column] = st.number_input(f'{column} (0 for Male, 1 for Female): ', min_value=0, max_value=1, format="%d")
        else:
            features_input[column] = st.number_input(f'{column}: ', min_value=0)

    # Button for processing the inputted raw data
    if st.button("Process Raw Data", key="process_raw_data"):
        # Create a DataFrame with the input data
        input_df = pd.DataFrame(features_input, index=[0])

        # Get probabilities of positive class using the pre-trained model
        predictions_proba = predict(input_df[required_columns], model)
        threshold = 0.5  # Set the threshold for detection to 50%

        # Convert numpy array to Pandas Series
        predictions_proba_series = pd.Series(predictions_proba)

        # Convert the elements to float and fill NaN values with 0
        predictions_proba_numeric = pd.to_numeric(predictions_proba_series, errors='coerce').fillna(0)

        # Convert probabilities to binary predictions using the threshold
        predictions = (predictions_proba_numeric.astype(float) > threshold).astype(int)

        st.subheader("Final Results:")
        st.write("Pancreatic Cancer Detected" if any(predictions) else "Not Detected")


