import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
best_model = XGBRegressor()
best_model.load_model("best_xgb_model.json")

# Load the scaler
scaler = joblib.load("scaler.pkl")

# Define the list of predictors
predictors = [
    'ArrivalInterval', 'EntitiesInBlockQ_U01', 'EntitiesInBlockWS_U01',
    'EntitiesInBlockQ_U02', 'EntitiesInBlockWS_U02', 'EntitiesInBlockQ_U03',
    'EntitiesInBlockWS_U03', 'EntitiesInBlockQ_U04', 'EntitiesInBlockWS_U04',
    'EntitiesInBlockQ_U05', 'EntitiesInBlockWS_U05', 'EntitiesInBlockQ_U07',
    'EntitiesInBlockWS_U07', 'EntitiesInBlockQ_U08', 'EntitiesInBlockWS_U08',
    'EntitiesInBlockQ_U11', 'EntitiesInBlockWS_U11', 'EntitiesInBlockQ_U13',
    'EntitiesInBlockWS_U13', 'EntitiesInBlockQ_U50', 'EntitiesInBlockWS_U50',
    'EntitiesInBlockQ_U60', 'EntitiesInBlockWS_U60', 'EntitiesInBlockQ_U70',
    'EntitiesInBlockWS_U70', 'EntitiesInBlockQ_UA1', 'EntitiesInBlockWS_UA1',
    'EntitiesInBlockQ_UD1', 'EntitiesInBlockWS_UD1', 'EntitiesInBlockQ_UB1',
    'EntitiesInBlockWS_UB1'
]

queue_predictors = [p for p in predictors if p.startswith('EntitiesInBlockQ_')]
wip_predictors = [p for p in predictors if p.startswith('EntitiesInBlockWS_')]

# Streamlit UI
st.markdown("<h1 style='text-align: center; font-weight: bold;'>Orders Flow Time Simulator</h1>", unsafe_allow_html=True)

# Create the sliders and layout
inputs = {}

# Create the Arrival Interval slider in the top center
st.markdown("<h3 style='text-align: center; color: red;'>Arrival Interval (Hours)</h3>", unsafe_allow_html=True)
inputs['ArrivalInterval'] = st.slider('Arrival Interval (Hours)', min_value=0, max_value=200, value=100, step=1, key='ArrivalInterval')

# Create a 3-column layout
left_col, center_col, right_col = st.columns([1, 2, 1])

with left_col:
    st.markdown("### Queue Size (Hours)")
    for predictor in queue_predictors:
        key = f"Queue_{predictor}"
        inputs[predictor] = st.slider(f"{predictor.split('_')[-1]}", min_value=0, max_value=100, value=50, step=1, key=key)

with right_col:
    st.markdown("### WIP (Hours)")
    for predictor in wip_predictors:
        key = f"WIP_{predictor}"
        inputs[predictor] = st.slider(f"{predictor.split('_')[-1]}", min_value=0, max_value=100, value=50, step=1, key=key)

# Convert inputs to DataFrame using the original feature names
input_df = pd.DataFrame([inputs], columns=predictors)

# Scale the inputs
input_scaled = scaler.transform(input_df)

# Make prediction
prediction = best_model.predict(input_scaled)

# Display the Flow Time in a big circle
center_col.markdown(f"""
<div style="display: flex; justify-content: center; align-items: center; height: 300px; width: 300px; border-radius: 50%; background-color: lightblue; margin: auto;">
    <h1 style="text-align: center;">{prediction[0]:.2f} Hours</h1>
</div>
""", unsafe_allow_html=True)
