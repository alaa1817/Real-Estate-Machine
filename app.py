import streamlit as st
import pandas as pd
import numpy as np
import joblib
from openai import OpenAI
import os

# =========================
# Load Saved Files
# =========================
regression_model = joblib.load("regression_model.pkl")
classifier_model = joblib.load("classifier_model.pkl")
model_columns = joblib.load("model_columns.pkl")
scaler = joblib.load("scaler.pkl")
num_cols = joblib.load("num_cols.pkl")

# =========================
# Groq Client
# =========================
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# =========================
# LLM Function
# =========================
def get_llm_explanation(price, category, bedrooms, bathrooms, sqft, waterfront, grade):
    prompt = f"""
    A house has the following details:

    Predicted Price: {price:.2f} USD
    Category: {category}

    Bedrooms: {bedrooms}
    Bathrooms: {bathrooms}
    Living Area: {sqft} sqft
    Waterfront: {waterfront}
    Grade: {grade}

    Explain clearly why the model predicted this price.
    Also explain why it was classified as {category}.
    """
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Could not fetch LLM explanation: {e}"

# =========================
# Streamlit UI
# =========================
st.title("🏠 AI Real Estate Price Prediction System")
st.write("Enter house details below:")

bedrooms = st.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Bathrooms", 1, 10, 2)
sqft = st.number_input("Living Area (sqft)", value=1500)
waterfront = st.selectbox("Waterfront", [0, 1])
grade = st.slider("House Grade", 1, 13, 7)

if st.button("Predict"):

    # -------------------------
    # Create Input DataFrame
    # -------------------------
    input_dict = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft,
        "waterfront": waterfront,
        "grade": grade
    }
    input_df = pd.DataFrame([input_dict])

    # -------------------------
    # Ensure all training columns exist
    # -------------------------
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Fill missing columns with 0

    # -------------------------
    # Reorder columns exactly as model expects
    # -------------------------
    input_df = input_df[model_columns]

    # -------------------------
    # Apply scaling safely
    # -------------------------
    # Use only the columns the scaler expects
    scaler_cols = [col for col in scaler.feature_names_in_ if col in input_df.columns]
    input_df[scaler_cols] = input_df[scaler_cols].astype(float)
    if scaler_cols:
        input_df[scaler_cols] = scaler.transform(input_df[scaler_cols])

    # -------------------------
    # Make Predictions
    # -------------------------
    try:
        predicted_price = regression_model.predict(input_df)[0]
        category_value = classifier_model.predict(input_df)[0]
        category = "High Price" if category_value == 1 else "Low Price"
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # -------------------------
    # LLM Explanation
    # -------------------------
    explanation = get_llm_explanation(
        predicted_price,
        category,
        bedrooms,
        bathrooms,
        sqft,
        waterfront,
        grade
    )

    # -------------------------
    # Display Results
    # -------------------------
    st.subheader("📊 Prediction Results")
    st.write(f"Predicted Price: ${predicted_price:,.2f}")
    st.write(f"Category: {category}")

    st.subheader("🧠 AI Explanation")
    st.write(explanation)
