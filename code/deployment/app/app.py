import streamlit as st
import requests
import json
import os

st.title('California Housing Price Prediction')

st.write('Enter the features of the house to predict its price:')

MedInc = st.number_input('Income (in $10,000)', min_value=0.0, max_value=15.0, value=5.0)
HouseAge = st.number_input('House Age (in years)', min_value=0, max_value=100, value=30, step=1)
AveRooms = st.number_input('Rooms', min_value=0.0, max_value=10.0, value=5.0, step=1.0)
AveBedrms = st.number_input('Bedrooms', min_value=0.0, max_value=5.0, value=1.0, step=1.0)
Population = st.number_input('Population (number of people within a block)', min_value=0, max_value=10000, value=1000, step=1)
AveOccup = st.number_input('Occupancy (average number of people living in 1 household within a block)', min_value=0.0, max_value=10.0, value=3.0, step=1.0)
Latitude = st.number_input('Latitude', min_value=32.0, max_value=42.0, value=35.0)
Longitude = st.number_input('Longitude', min_value=-124.0, max_value=-114.0, value=-119.0)

if st.button('Predict Price'):
    features = {
        'MedInc': MedInc,
        'HouseAge': HouseAge,
        'AveRooms': AveRooms,
        'AveBedrms': AveBedrms,
        'Population': Population,
        'AveOccup': AveOccup,
        'Latitude': Latitude,
        'Longitude': Longitude
    }
    
    api_host = os.getenv('API_HOST', 'api')
    api_port = os.getenv('API_PORT', '8000')
    response = requests.post(f'http://{api_host}:{api_port}/predict', json=features)
    
    if response.status_code == 200:
        prediction = response.json()['prediction']
        formatted_prediction = f"${prediction * 100000:,.0f}"
        st.success(f'The predicted house price is: {formatted_prediction}.')
    else:
        st.error('An error occurred while making the prediction.')