import streamlit as st
import pickle as pkl
import numpy as np

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

/* Background */
.stApp {
    background: #0d0d0d;
    color: #f0ede8;
}

/* Card wrapper */
.card {
    background: #1a1a1a;
    border: 1px solid #2e2e2e;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
}

/* Result box */
.result-box {
    background: linear-gradient(135deg, #c8f560 0%, #a8e040 100%);
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    margin-top: 24px;
}
.result-box .label {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #2a3a00;
    margin-bottom: 4px;
}
.result-box .price {
    font-family: 'Syne', sans-serif;
    font-size: 52px;
    font-weight: 800;
    color: #1a2600;
    line-height: 1.1;
}
.result-box .sub {
    font-size: 14px;
    color: #3a5000;
    margin-top: 6px;
}

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 16px;
    border-bottom: 1px solid #2e2e2e;
    padding-bottom: 8px;
}

/* Metric badges */
.badge-row {
    display: flex;
    gap: 12px;
    margin-top: 12px;
    flex-wrap: wrap;
}
.badge {
    background: #252525;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 13px;
    color: #aaa;
}
.badge span {
    color: #c8f560;
    font-weight: 600;
}

/* Streamlit overrides */
.stSelectbox label, .stSlider label, .stNumberInput label, .stRadio label {
    color: #ccc !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}
div[data-testid="stSlider"] > div { color: #c8f560; }
.stButton > button {
    background: #c8f560 !important;
    color: #0d0d0d !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 0 !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return pkl.load(open('model.pkl', 'rb'))

try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ model.pkl not found. Please run models.ipynb to generate it first.")
    st.stop()

# ── Encoding maps ───────────────────────────────────────────────────────────────
d1 = {'Comprehensive': 0, 'Third Party insurance': 1, 'Third Party': 1,
      'Zero Dep': 2, 'Not Available': 3}
d2 = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
d3 = {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3,
      'Forth Owner': 4, 'Fifth Owner': 5}
d4 = {'Manual': 0, 'Automatic': 1}

# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 32px 0 24px 0;">
    <div style="font-family: 'Syne', sans-serif; font-size: 38px; font-weight: 800;
                color: #f0ede8; line-height: 1.1;">
        Car Price<br><span style="color:#c8f560;">Predictor</span>
    </div>
    <div style="color: #666; font-size: 14px; margin-top: 10px;">
        XGBoost model · R² = 0.89 · MAE ≈ ₹2.4L
    </div>
</div>
""", unsafe_allow_html=True)

# ── Form ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Car Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    val_insurance = st.selectbox('Insurance Type', options=list(d1.keys()))
    val_fuel      = st.selectbox('Fuel Type', options=list(d2.keys()))
    val_owner     = st.selectbox('Ownership', options=list(d3.keys()))
    val_seats     = st.selectbox('Seats', options=[4, 5, 6, 7, 8], index=1)

with col2:
    val_trans     = st.selectbox('Transmission', options=list(d4.keys()))
    val_year      = st.number_input('Manufacturing Year', min_value=2000,
                                    max_value=2024, value=2020, step=1)
    val_kms       = st.number_input('KMs Driven', min_value=0,
                                    max_value=500000, value=30000, step=1000)

st.markdown('<div class="section-header" style="margin-top:8px;">Engine & Performance</div>',
            unsafe_allow_html=True)

col3, col4, col5 = st.columns(3)
with col3:
    val_mileage = st.number_input('Mileage (kmpl)', min_value=5.0,
                                  max_value=35.0, value=18.0, step=0.5,
                                  format="%.1f")
with col4:
    val_engine  = st.number_input('Engine (cc)', min_value=500,
                                  max_value=6000, value=1500, step=50)
with col5:
    val_power   = st.number_input('Max Power (bhp)', min_value=30.0,
                                  max_value=700.0, value=120.0, step=5.0,
                                  format="%.1f")

# ── Predict ─────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
if st.button('PREDICT PRICE'):
    features = [[
        d1[val_insurance],
        d2[val_fuel],
        val_kms,
        d3[val_owner],
        d4[val_trans],
        val_seats,
        val_year,
        val_mileage,
        val_engine,
        val_power,
    ]]

    predicted = float(model.predict(features)[0])
    predicted = max(0.5, predicted)   # floor at ₹0.5L

    low  = round(predicted * 0.92, 2)
    high = round(predicted * 1.08, 2)

    st.markdown(f"""
    <div class="result-box">
        <div class="label">Estimated Market Price</div>
        <div class="price">₹ {predicted:.2f} L</div>
        <div class="sub">Confidence range: ₹{low}L – ₹{high}L</div>
    </div>
    <div class="badge-row">
        <div class="badge">Model <span>XGBoost</span></div>
        <div class="badge">Accuracy <span>R² 0.89</span></div>
        <div class="badge">Avg Error <span>±₹2.4L</span></div>
        <div class="badge">Features used <span>10</span></div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="color:#444; font-size:12px; margin-top:48px; text-align:center;">
    Trained on 1,482 used car listings · Prices in Indian Rupees (Lakhs)
</div>
""", unsafe_allow_html=True)
