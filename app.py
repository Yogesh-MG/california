# app.py
import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np

# Load data
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression().fit(X_train, y_train)

st.title("🏠 California Housing Price Predictor")

# Input fields
inputs = []
for feature in X.columns:
    value = st.number_input(f"Enter {feature}", value=float(X[feature].mean()))
    inputs.append(value)

# Predict
if st.button("Predict Price"):
    result = model.predict([inputs])[0]
    st.success(f"Predicted Median House Value: ${round(result * 100000, 2)}")

    # Predict on test set and calculate metrics
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write("### Model Performance Metrics")
    st.write(f"Mean Squared Error: {mse:.4f}")
    st.write(f"Mean Absolute Error: {mae:.4f}")
    st.write(f"R² Score: {r2:.4f}")
