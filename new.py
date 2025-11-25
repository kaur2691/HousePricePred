import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS (NEW MODERN WHITE + TEAL THEME)
# ============================================================================
st.markdown("""
<style>

/* --- MAIN APP BACKGROUND --- */
.stApp {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
}

/* --- MAIN CONTAINER / CARDS --- */
.block-container {
    background: rgba(255, 255, 255, 0.90);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

/* --- TITLES --- */
h1, h2, h3, h4, h5, h6 {
    color: #343a40 !important;
    font-family: 'Segoe UI', sans-serif;
    font-weight: 700;
}

/* --- SIDEBAR --- */
.css-1d391kg {
    background-color: #ffffff !important;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

/* --- BUTTON STYLING (Teal Gradient) --- */
.stButton>button {
    background: linear-gradient(135deg, #17a2b8 0%, #138496 100%);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 15px 40px;
    font-size: 18px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(23, 162, 184, 0.4);
}

/* --- INPUT FIELDS --- */
.stNumberInput>div>div>input, 
.stSelectbox>div>div>select {
    border-radius: 8px !important;
    border: 1px solid #17a2b8 !important;
}

/* --- METRIC CARDS --- */
.css-1xarl3l {
    background: #ffffff !important;
    border-left: 5px solid #17a2b8 !important;
    border-radius: 12px !important;
    padding: 20px !important;
    color: #343a40 !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

/* --- SUCCESS BOX --- */
.stSuccess {
    background-color: #d1f0f4 !important;
    color: #0c5460 !important;
    border-left: 6px solid #17a2b8;
    border-radius: 10px;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND SCALER
# ============================================================================
@st.cache_resource
def load_model_and_scaler():
    try:
        with open('house_price_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler, True
    except FileNotFoundError:
        return None, None, False

model, scaler, model_loaded = load_model_and_scaler()

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def predict_house_price(bedrooms, grade, has_basement, living_in_m2, renovated, 
                        nice_view, perfect_condition, real_bathrooms, has_lavatory, 
                        single_floor, month, quartile_zone):

    input_data = pd.DataFrame({
        'bedrooms': [bedrooms],
        'grade': [grade],
        'has_basement': [int(has_basement)],
        'living_in_m2': [living_in_m2],
        'renovated': [int(renovated)],
        'nice_view': [int(nice_view)],
        'perfect_condition': [int(perfect_condition)],
        'real_bathrooms': [real_bathrooms],
        'has_lavatory': [int(has_lavatory)],
        'single_floor': [int(single_floor)],
        'month': [month],
        'quartile_zone': [quartile_zone]
    })

    input_data['bathroom_bedroom_ratio'] = input_data['real_bathrooms'] / (input_data['bedrooms'] + 1)
    input_data['total_rooms'] = input_data['bedrooms'] + input_data['real_bathrooms']
    input_data['luxury_score'] = (input_data['grade'] * input_data['living_in_m2']) / 100
    input_data['quality_index'] = input_data['grade'] * (input_data['perfect_condition'] + 1)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    return prediction, input_data

# ============================================================================
# HEADER
# ============================================================================
st.markdown("<h1 style='text-align: center;'>🏠 House Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color:#495057;'>Estimate the value of your property with AI-powered insights</p>", unsafe_allow_html=True)
st.markdown("---")

# ============================================================================
# CHECK IF MODEL IS LOADED
# ============================================================================
if not model_loaded:
    st.error("⚠️ Model files not found! Please add 'house_price_model.pkl' and 'scaler.pkl'.")
    st.stop()

# ============================================================================
# SIDEBAR INPUTS
# ============================================================================
st.sidebar.header("🏡 Property Details")
st.sidebar.markdown("---")

bedrooms = st.sidebar.selectbox("Bedrooms", [1,2,3], index=1)
real_bathrooms = st.sidebar.selectbox("Bathrooms", [1,2,3], index=1)
living_in_m2 = st.sidebar.slider("Living Area (m²)", 49.0, 400.0, 180.0, 5.0)

st.sidebar.markdown("---")
st.sidebar.subheader(" Quality")

grade = st.sidebar.select_slider("Property Grade", [1,2,3,4,5], value=3)
quartile_zone = st.sidebar.select_slider("Zone Quality", [1,2,3,4], value=2)

st.sidebar.markdown("---")
st.sidebar.subheader(" Features")

col1, col2 = st.sidebar.columns(2)

with col1:
    has_basement = st.checkbox("Basement")
    renovated = st.checkbox("Renovated")
    nice_view = st.checkbox("Nice View")

with col2:
    perfect_condition = st.checkbox("Perfect Condition")
    has_lavatory = st.checkbox("Extra Lavatory")
    single_floor = st.checkbox("Single Floor")

st.sidebar.markdown("---")
month = st.sidebar.selectbox("Month of Sale", list(range(1,13)),
                             format_func=lambda x: datetime(2024, x, 1).strftime('%B'))

# ============================================================================
# MAIN AREA PREDICTION
# ============================================================================
col1, col2, col3 = st.columns([1,2,1])

with col2:
    predict = st.button("Predict House Price", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

if predict:
    predicted_price, input_df = predict_house_price(
        bedrooms, grade, has_basement, living_in_m2, renovated, nice_view,
        perfect_condition, real_bathrooms, has_lavatory, single_floor,
        month, quartile_zone
    )

    st.success("Prediction Completed!")

    # PRICE CARD
    st.markdown(f"""
        <div style='background:#ffffff;
                    padding:35px;
                    border-radius:20px;
                    text-align:center;
                    border: 2px solid #17a2b8;
                    box-shadow:0 4px 15px rgba(0,0,0,0.07);'>
            <h2 style='color:#343a40;'>Estimated Price</h2>
            <h1 style='color:#17a2b8; font-size:48px;'>${predicted_price:,.2f}</h1>
            <p style='color:#6c757d;'>Based on AI & Market Factors</p>
        </div>
    """, unsafe_allow_html=True)

else:
    st.info("👉 Fill details on the left and click **Predict House Price**")

