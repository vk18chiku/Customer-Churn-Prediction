import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pandas as pd
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .css-1d391kg {
        background-color: rgba(255, 255, 255, 0.05);
    }
    h1 {
        color: #ffffff;
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #e0e0e0;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .prediction-box {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin-top: 2rem;
    }
    .metric-container {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    </style>
""", unsafe_allow_html=True)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# load the trained data
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(os.path.join(script_dir, 'model.h5'))

@st.cache_resource
def load_encoders():
    with open(os.path.join(script_dir, 'label_encoder_gender.pkl'), 'rb') as file:
        label_encoder_gender = pickle.load(file)
    with open(os.path.join(script_dir, 'onehot_encoder_geo.pkl'), 'rb') as file:
        onehot_encoder_geo = pickle.load(file)
    with open(os.path.join(script_dir, 'scaler.pkl'), 'rb') as file:
        scaler = pickle.load(file)
    return label_encoder_gender, onehot_encoder_geo, scaler

model = load_model()
label_encoder_gender, onehot_encoder_geo, scaler = load_encoders()

# Header
st.title('🔮 Customer Churn Prediction')
st.markdown('<p class="subtitle">Predict customer churn probability using AI-powered analytics</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📋 About")
    st.info("""
    This application uses a Neural Network model to predict the likelihood 
    of customer churn based on various customer attributes.
    
    **Model Accuracy:** High precision ANN model
    
    **Features Used:**
    - Customer Demographics
    - Account Information
    - Banking Behavior
    """)
    
    st.header("🎯 How to Use")
    st.markdown("""
    1. Fill in customer details
    2. Click 'Predict Churn'
    3. View prediction results
    """)

# Main content area
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("### 👤 Customer Information")
    
    # Create two columns for inputs
    input_col1, input_col2 = st.columns(2)
    
    with input_col1:
        geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
        gender = st.selectbox('👥 Gender', label_encoder_gender.classes_)
        age = st.slider('🎂 Age', 18, 92, 35)
        credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=850, value=650)
        balance = st.number_input('💰 Balance', min_value=0.0, value=50000.0, step=1000.0)
    
    with input_col2:
        estimated_salary = st.number_input('💵 Estimated Salary', min_value=0.0, value=50000.0, step=1000.0)
        tenure = st.slider('📅 Tenure (years)', 0, 10, 5)
        num_of_products = st.slider('🛍️ Number of Products', 1, 4, 2)
        has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        is_active_member = st.selectbox('✅ Is Active Member', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    
    # Predict button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button('🔮 Predict Churn Probability'):
        # Prepare the input data
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })

        # One-hot encode 'Geography'
        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out())

        # Combine one-hot encoded columns with input data
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Predict churn
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]
        
        # Display results with styled container
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        
        # Create three columns for metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric(label="Churn Probability", value=f"{prediction_proba:.1%}")
        
        with metric_col2:
            st.metric(label="Confidence", value=f"{abs(prediction_proba - 0.5) * 200:.1f}%")
        
        with metric_col3:
            risk_level = "🔴 High" if prediction_proba > 0.7 else "🟡 Medium" if prediction_proba > 0.4 else "🟢 Low"
            st.metric(label="Risk Level", value=risk_level)
        
        # Progress bar
        st.progress(float(prediction_proba))
        
        # Final verdict
        st.markdown("<br>", unsafe_allow_html=True)
        if prediction_proba > 0.5:
            st.error(f"""
            ### ⚠️ High Churn Risk
            This customer has a **{prediction_proba:.1%}** probability of churning. 
            Consider retention strategies such as personalized offers or customer engagement programs.
            """)
        else:
            st.success(f"""
            ### ✅ Low Churn Risk
            This customer has a **{(1-prediction_proba):.1%}** probability of staying. 
            Continue providing excellent service to maintain satisfaction.
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #e0e0e0;'>
    <p>Powered by TensorFlow & Streamlit | AI-Driven Customer Analytics</p>
</div>
""", unsafe_allow_html=True)