import streamlit as st
import pandas as pd
import numpy as np
import joblib
from openai import OpenAI
import os

# =========================
# Load models and preprocessing files
# =========================
regression_model = joblib.load("regression_model.pkl")
classifier_model = joblib.load("classifier_model.pkl")
model_columns = joblib.load("model_columns.pkl")
scaler = joblib.load("scaler.pkl")
num_cols = joblib.load("num_cols.pkl")

# =========================
# Extract city names dynamically
# =========================
city_columns = [col for col in model_columns if col.startswith("city_")]
cities = [col.replace("city_", "") for col in city_columns]

# =========================
# LLM Client
# =========================
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def get_llm_explanation(price, category, city):
    prompt = f"""
    A house price prediction system predicted:

    Price: {price:.2f} USD
    Category: {category}
    City: {city}

    Explain clearly why this prediction makes sense based on real estate factors such as:
    - location (city)
    - size (sqft)
    - number of rooms
    - condition
    Keep it simple and professional.
    """
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

# =========================
# Streamlit UI
# =========================
st.title("🏠 AI Real Estate Prediction")

# Inputs
bedrooms = st.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Bathrooms", 1, 10, 2)
sqft_living = st.number_input("Living Area", 100, 10000, 1500)
sqft_lot = st.number_input("Lot Area", 500, 100000, 5000)
floors = st.number_input("Floors", 1, 5, 1)
waterfront = st.selectbox("Waterfront", [0, 1])
view = st.slider("View", 0, 4, 0)
condition = st.slider("Condition", 1, 5, 3)
sqft_above = st.number_input("Sqft Above", 100, 10000, 1200)
sqft_basement = st.number_input("Sqft Basement", 0, 5000, 300)
yr_built = st.number_input("Year Built", 1900, 2025, 2000)
yr_renovated = st.number_input("Year Renovated", 0, 2025, 0)
year = st.number_input("Sale Year", 2000, 2025, 2014)
month = st.slider("Month", 1, 12, 6)
day = st.slider("Day", 1, 31, 15)

# ✅ City Input (dynamic)
city = st.selectbox("City", cities)

# =========================
# Prediction
# =========================
if st.button("Predict"):
    # 1️⃣ Prepare input
    input_dict = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft_living': sqft_living,
        'sqft_lot': sqft_lot,
        'floors': floors,
        'waterfront': waterfront,
        'view': view,
        'condition': condition,
        'sqft_above': sqft_above,
        'sqft_basement': sqft_basement,
        'yr_built': yr_built,
        'yr_renovated': yr_renovated,
        'year': year,
        'month': month,
        'day': day
    }

    input_df = pd.DataFrame([input_dict])

    # 2️⃣ Add city as one-hot
    city_col = f"city_{city}"
    input_df[city_col] = 1

    # 3️⃣ Match training columns
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # 4️⃣ Scaling
    valid_num_cols = [col for col in num_cols if col in input_df.columns]
    input_df[valid_num_cols] = scaler.transform(input_df[valid_num_cols])

    # 5️⃣ Predictions
    predicted_price_log = regression_model.predict(input_df)
    predicted_price = np.expm1(predicted_price_log)[0]

    category_value = classifier_model.predict(input_df)[0]
    category = "High Price" if category_value == 2 else "Medium Price" if category_value == 1 else "Low Price"

    # 6️⃣ LLM Explanation
    explanation = get_llm_explanation(predicted_price, category, city)

    # =========================
    # Output
    # =========================
    st.subheader("📊 Results")
    st.write(f"Predicted Price: ${predicted_price:,.2f}")
    st.write(f"Category: {category}")

    st.subheader("🧠 AI Explanation")
    st.write(explanation)
