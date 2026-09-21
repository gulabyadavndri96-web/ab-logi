import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('logi.sav')

st.title('Delivery Delay Prediction App')
st.write('Enter the features below to predict if there will be a delivery delay.')

# Define the feature names in the correct order as used during training
feature_names = [
    'Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
    'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
    'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
    'Warehouse_Processing_Time'
]

# Create input fields for each feature
input_data = {}
input_data['Delivery_Distance'] = st.number_input('Delivery Distance (km)', min_value=0.0, value=12.0)
input_data['Traffic_Congestion'] = st.number_input('Traffic Congestion (1-5, 5 being highest)', min_value=1, max_value=5, value=2)
input_data['Weather_Condition'] = st.number_input('Weather Condition (1-5, 5 being worst)', min_value=1, max_value=5, value=1)
input_data['Delivery_Slot'] = st.number_input('Delivery Slot (1-3)', min_value=1, max_value=3, value=2)
input_data['Driver_Experience'] = st.number_input('Driver Experience (years)', min_value=0, value=4)
input_data['Num_Stops'] = st.number_input('Number of Stops', min_value=0, value=2)
input_data['Vehicle_Age'] = st.number_input('Vehicle Age (years)', min_value=0, value=3)
input_data['Road_Condition_Score'] = st.number_input('Road Condition Score (1-5, 5 being best)', min_value=1, max_value=5, value=3)
input_data['Package_Weight'] = st.number_input('Package Weight (kg)', min_value=0.0, value=12.0)
input_data['Fuel_Efficiency'] = st.number_input('Fuel Efficiency (km/L)', min_value=0.0, value=10.0)
input_data['Warehouse_Processing_Time'] = st.number_input('Warehouse Processing Time (minutes)', min_value=0, value=120)

# Convert input data to a Pandas DataFrame
# Ensure the order of columns matches the feature_names
input_df = pd.DataFrame([input_data], columns=feature_names)

if st.button('Predict Delivery Delay'):
    try:
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader('Prediction Results:')
        if prediction[0] == 1:
            st.error('**Delivery is likely to be Delayed!**')
        else:
            st.success('**Delivery is likely to be On Time.**')

        st.write(f'Probability of No Delay (Class 0): {prediction_proba[0][0]:.2f}')
        st.write(f'Probability of Delay (Class 1): {prediction_proba[0][1]:.2f}')

    except Exception as e:
        st.error(f'An error occurred during prediction: {e}')


# Instructions on how to run the app
st.sidebar.subheader('How to run this app:')
st.sidebar.write('1. Save the code above as `streamlit_app.py` in the same directory as `logi.sav`.')
st.sidebar.write('2. Make sure you have Streamlit installed: `pip install streamlit`')
st.sidebar.write('3. Open your terminal or command prompt.')
st.sidebar.write('4. Navigate to the directory where you saved the files.')
st.sidebar.write('5. Run the app using the command: `streamlit run streamlit_app.py`')
