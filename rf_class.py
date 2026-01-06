import streamlit as st
import numpy as np
import pandas as pd
import pickle

# 1️⃣ Load Random Forest model
with open('rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

# 2️⃣ Load LabelEncoder
with open('le.pkl', 'rb') as f:
    le = pickle.load(f)

st.title("🍎 Fruit Classification using Random Forest")
st.write("Enter the feature values to predict the fruit class")

# Input fields using actual feature names
size_cm = st.number_input("Size (cm)")
weight_g = st.number_input("Weight (g)")
avg_price = st.number_input("Average Price (₹)")

# For categorical features, use select boxes
shape = st.selectbox("Shape", ["round", "oval", "elongated"])  # replace with actual possible values
color = st.selectbox("Color", ["red", "green", "yellow"])      # replace with actual possible values
taste = st.selectbox("Taste", ["sweet", "sour", "bitter"])     # replace with actual possible values

if st.button("Predict"):
    # Create DataFrame for the input
    input_df = pd.DataFrame({
        'size (cm)': [size_cm],
        'weight (g)': [weight_g],
        'avg_price (₹)': [avg_price],
        'shape': [shape],
        'color': [color],
        'taste': [taste]
    })

    # Convert categorical features to numeric using get_dummies
    input_df = pd.get_dummies(input_df)

    # Ensure all columns match the model's training columns
    model_columns = model.feature_names_in_
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # add missing columns
    input_df = input_df[model_columns]  # reorder columns


    # Make prediction
    prediction = model.predict(input_df)

    # Convert numeric prediction back to original fruit name
    fruit_name = le.inverse_transform(prediction)
    st.success(f"Predicted Fruit: {fruit_name[0]}")
