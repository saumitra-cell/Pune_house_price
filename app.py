import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Pune House Price Predictor", page_icon="🏠")


@st.cache_resource(show_spinner="Loading model...")
def load():
    try:
        return joblib.load(Path("model/model.joblib")), json.loads(Path("model/meta.json").read_text())
    except Exception:
        # Saved model missing or built with a different scikit-learn: retrain from the data
        import train
        train.main()
        return joblib.load(Path("model/model.joblib")), json.loads(Path("model/meta.json").read_text())


model, meta = load()
mae = meta["results"][meta["best_model"]]["test_mae_lakhs"]

st.title("🏠 Pune House Price Predictor")
st.caption("Estimate a property's price from its size, rooms and location.")

col1, col2 = st.columns(2)
with col1:
    sqft = st.number_input("Total area (sq ft)", 300, 10000, 1000, step=50)
    bhk = st.number_input("Bedrooms (BHK)", 1, 8, 2)
with col2:
    bath = st.number_input("Bathrooms", 1, 8, 2)
    balcony = st.number_input("Balconies", 0, 4, 1)

locations = [l for l in meta["locations"] if l != "other"] + ["other"]
location = st.selectbox("Location", locations)

if st.button("Predict price", type="primary"):
    row = pd.DataFrame([{
        "total_sqft": sqft, "bhk": bhk, "bath": bath,
        "balcony": balcony, "location": location,
    }])
    price = float(model.predict(row)[0])
    shown = f"₹ {price / 100:.2f} crore" if price >= 100 else f"₹ {price:.1f} lakh"
    st.success(f"Estimated price: **{shown}**")
    st.caption(
        f"Typical error of this model is about ₹ {mae:.0f} lakh. "
        "Treat this as a rough estimate, not a valuation."
    )

with st.expander("About this model"):
    st.write(f"Best model: **{meta['best_model']}**")
    st.table(pd.DataFrame(meta["results"]).T)
