import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Airbnb Price Predictor", layout="centered")
st.title("🏠 Airbnb Price Predictor (10 Features)")

@st.cache_resource
def load_all():
    model = load_model("model.h5")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_all()

st.markdown("Enter the Airbnb listing details:")

col1, col2 = st.columns(2)
with col1:
    lat = st.number_input("Latitude", value=43.65)
    long = st.number_input("Longitude", value=-79.38)
    construction_year = st.number_input("Construction Year", min_value=1800, max_value=2025, value=2000)
    service_fee = st.number_input("Service Fee ($)", min_value=0.0, value=50.0)
    min_nights = st.number_input("Minimum Nights", min_value=1, value=3)

with col2:
    num_reviews = st.number_input("Number of Reviews", min_value=0, value=10)
    reviews_per_month = st.number_input("Reviews per Month", min_value=0.0, value=1.2)
    review_rate = st.slider("Review Rating", 0.0, 5.0, 4.5, 0.1)
    host_listings_count = st.number_input("Host Listings Count", min_value=1, value=2)
    availability = st.slider("Availability (days/year)", 0, 365, 200)

if st.button("Predict Price 💰"):
    try:
        input_data = np.array([[
            lat,
            long,
            construction_year,
            service_fee,
            min_nights,
            num_reviews,
            reviews_per_month,
            review_rate,
            host_listings_count,
            availability
        ]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0][0]
        st.success(f"Estimated Airbnb Price: ${prediction:.2f}")
    except Exception as e:
        st.error(f"❌ Error: {e}")
