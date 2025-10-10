import streamlit as st
import pandas as pd
import pickle
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Smartphone Price Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Model and Data ---
@st.cache_data
def load_model_and_data():
    """Loads the trained pipeline and the dataset for UI options."""
    try:
        # Ensure you use the name of the saved file from your hyper-tuning script
        with open('random_forest_model.pkl', 'rb') as file:
            pipeline = pickle.load(file)
    except FileNotFoundError:
        st.error("Model file ('random_forest_model.pkl') not found. Please run the training script first.")
        return None, None
        
    df = pd.read_csv('flipkart_smartphones_cleaned.csv')
    return pipeline, df

ml_pipeline, df = load_model_and_data()

# --- Helper Function ---
def get_unique_values(feature_name):
    """Gets a sorted list of unique values for a given feature."""
    if df is not None:
        return sorted(df[feature_name].dropna().unique().tolist())
    return []

# --- Custom CSS Styling ---
st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #0f1117;
    color: #e0e0e0;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #161a24;
    padding: 1.5rem 1rem;
    border-right: 1px solid #262b3a;
}

[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #ffffff;
}

/* Inputs */
div[data-baseweb="select"] > div {
    background-color: #1e2230 !important;
    color: #ffffff !important;
}

.stSlider > div {
    background-color: transparent !important;
}

.css-1n76uvr, .stNumberInput input, .stTextInput input {
    background-color: #1e2230 !important;
    color: #ffffff !important;
}

/* Divider line */
.sidebar-divider {
    border-top: 1px solid #333a4d;
    margin: 15px 0 25px 0;
}

/* Centered Predict Button */
.center-button {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: 10px;
}

/* Gradient Predict Price button */
.stButton>button {
    background: linear-gradient(135deg, #0078ff, #00c6ff);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    font-size: 16px;
    padding: 10px 25px;
    width: auto;
    transition: all 0.3s ease;
    box-shadow: 0 4px 10px rgba(0, 120, 255, 0.3);
}

.stButton>button:hover {
    background: linear-gradient(135deg, #00b4ff, #0078ff);
    box-shadow: 0 6px 14px rgba(0, 120, 255, 0.5);
    transform: scale(1.05);
}

/* Main content */
h1, h2, h3, h4 {
    color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
}

/* Prediction result card */
.prediction-box {
    background-color: #161a24;
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 0 15px rgba(0,0,0,0.4);
    margin-top: 20px;
}

/* Disclaimer bar */
.disclaimer-bar {
    background-color: #1e3a55;
    color: #d0d0d0;
    padding: 15px;
    border-radius: 8px;
    margin-top: 15px;
    text-align: center;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar Inputs ---
if df is not None:
    with st.sidebar:
        st.markdown(
    """
    <div style="text-align: center; padding: 10px 0;">
        <h2>Phone Specs</h2>
        <p>Set your phone’s specs to see its estimated price.</p>
        <hr style="border:1px solid #eee;">
    </div>
    """,
    unsafe_allow_html=True
)

        # Categorical Features
        brand = st.selectbox("Brand", options=get_unique_values('Brand'))
        processor = st.selectbox("Processor", options=get_unique_values('Processor'))
        display_type = st.selectbox("Display Type", options=get_unique_values('Display_Type'))

        # Numerical Features
        ram_gb = st.select_slider("RAM (GB)", options=[2, 3, 4, 6, 8, 12, 16], value=8)
        rom_gb = st.select_slider("Storage (GB)", options=[32, 64, 128, 256, 512, 1024], value=128)
        display_size = st.slider("Display Size (inches)", 5.0, 7.2, 6.5, 0.1)
        battery_mah = st.slider("Battery (mAh)", 2500, 7500, 5000, 100)
        warranty = st.select_slider("Warranty (Years)", options=[0, 1, 2], value=1)

        st.markdown('<div class="center-button">', unsafe_allow_html=True)
        predict_button = st.button("🎯 Predict Price")
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.warning("Data could not be loaded. Please check the file path and try again.")
    predict_button = False

# --- Main Section ---
st.markdown("## 📱 Smartphone Price Predictor")
st.write("Provide smartphone specifications in the sidebar to get an **AI-estimated price**.")
st.markdown("---")

# --- Prediction Logic ---
if predict_button and ml_pipeline is not None:
    input_data = pd.DataFrame({
        'Brand': [brand], 
        'RAM_GB': [ram_gb], 
        'ROM_GB': [rom_gb],
        'Display_Size_inch': [display_size], 
        'Display_Type': [display_type],
        'Battery_mAh': [battery_mah], 
        'Processor': [processor], 
        'Warranty_Years': [warranty]
    })

    with st.spinner("⚙️ Predicting best price..."):
        time.sleep(1) # Small delay for better UX
        prediction = ml_pipeline.predict(input_data)
        predicted_price = int(prediction[0])

    st.markdown(f"""
    <div class="prediction-box">
        <h2>Predicted Smartphone Price</h2>
        <h1 style="color:#00b4ff;">₹ {predicted_price:,.0f}</h1>
        <p style="color:#aaaaaa;">Estimated based on a fine-tuned Machine Learning model</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-bar">
        Disclaimer : The estimated price is generated using a machine learning model trained on real-time Flipkart smartphone listings collected on 8 October, 2025 11:45 AM.
Since online prices fluctuate frequently due to seasonal sales, promotional offers, regional variations, and market demand, the actual price may differ from this estimate.
This prediction is intended to provide a data-driven price approximation, not an exact market value.
    </div>
    """, unsafe_allow_html=True)

# --- Explanatory Section ---
st.markdown("<br><br>", unsafe_allow_html=True)
with st.expander("📘 Explore Project Details", expanded=False):
    # --- UPDATED: Using the more detailed project description we created ---
    st.markdown("""
    This application represents a end-to-end data science project — covering data collection, preprocessing, model training, hyperparameter tuning, and deployment.

    #### **1. Data Sourcing & Preparation**
    - **Data Collection —** The dataset was built by scraping real-time smartphone listings from **Flipkart**, ensuring the model is trained on data that reflects current market conditions.
    - **Data Cleaning & Feature Engineering —** Key specifications like Brand, RAM, Storage, Processor, and Battery were extracted and standardized from raw product descriptions to create a clean, model-ready dataset.

    #### **2. Model Development & Optimization**
    - **Algorithm Selection —** The prediction model is a `RandomForestRegressor`, an ensemble algorithm chosen for its high accuracy and robustness against overfitting.
    - **Hyperparameter Tuning —** To achieve the best possible performance, the model underwent extensive hyperparameter tuning using **`RandomizedSearchCV`**. This process systematically tested numerous parameter combinations to find the optimal settings for maximum predictive accuracy.
    - **Training —** The final, optimized model was trained on a curated dataset of over 900 unique smartphone records, allowing it to learn the complex relationships between device features and market price.

    #### **3. Workflow & Performance**
    - **Automated ML Pipeline —** A complete `scikit-learn` Pipeline automates the entire workflow. It seamlessly handles data preprocessing (scaling numerical data and one-hot encoding categorical data) and feeds the results into the model for prediction, ensuring consistency and reproducibility.
    - **Final Performance —** The tuned model achieves a **Mean Absolute Error (MAE) of approximately ₹1,430**, meaning its price predictions are, on average, extremely close to the actual listing price.
    """)
