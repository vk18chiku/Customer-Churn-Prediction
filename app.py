import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetic design
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #667eea;
        text-align: center;
        font-size: 3rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #667eea !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
    }
    .prediction-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: 800;
        margin: 1rem 0;
    }
    .prediction-message {
        font-size: 1.3rem;
        font-weight: 500;
    }
    .info-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Header
st.title('🎯 Customer Churn Predictor')
st.markdown('<p class="subtitle">Predict customer churn probability with advanced machine learning</p>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👤 Personal Information")
    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('⚧ Gender', label_encoder_gender.classes_)
    age = st.slider('🎂 Age', 18, 92, 35)
    
    st.markdown("### 💼 Account Details")
    balance = st.number_input('💰 Balance', min_value=0.0, value=50000.0, step=1000.0)
    credit_score = st.number_input('📊 Credit Score', min_value=300, max_value=850, value=650, step=10)

with col2:
    st.markdown("### 💵 Financial Information")
    estimated_salary = st.number_input('💸 Estimated Salary', min_value=0.0, value=50000.0, step=1000.0)
    tenure = st.slider('📅 Tenure (years)', 0, 10, 5)
    num_of_products = st.slider('🛍️ Number of Products', 1, 4, 2)
    
    st.markdown("### ✅ Account Status")
    has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    is_active_member = st.selectbox('⭐ Is Active Member', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# Predict button
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
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine one-hot encoded columns with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict churn
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]
    
    # Display results with aesthetic styling
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    if prediction_proba > 0.5:
        st.markdown(f"""
            <div class="prediction-box" style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);">
                <div class="prediction-title">⚠️ CHURN ALERT</div>
                <div class="prediction-value">{prediction_proba:.1%}</div>
                <div class="prediction-message">The customer is likely to churn</div>
                <p style="margin-top: 1rem; font-size: 1rem; opacity: 0.9;">
                    High risk of customer leaving. Consider retention strategies.
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="prediction-box" style="background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);">
                <div class="prediction-title">✅ RETENTION LIKELY</div>
                <div class="prediction-value">{prediction_proba:.1%}</div>
                <div class="prediction-message">The customer is likely to stay</div>
                <p style="margin-top: 1rem; font-size: 1rem; opacity: 0.9;">
                    Low risk of customer leaving. Continue excellent service.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Additional insights
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📈 Customer Insights")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown(f"""
            <div class="info-card">
                <h4 style="color: #667eea; margin-bottom: 0.5rem;">💰 Balance</h4>
                <p style="font-size: 1.5rem; font-weight: 700; color: #333; margin: 0;">${balance:,.2f}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col_b:
        st.markdown(f"""
            <div class="info-card">
                <h4 style="color: #667eea; margin-bottom: 0.5rem;">📊 Credit Score</h4>
                <p style="font-size: 1.5rem; font-weight: 700; color: #333; margin: 0;">{credit_score}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col_c:
        st.markdown(f"""
            <div class="info-card">
                <h4 style="color: #667eea; margin-bottom: 0.5rem;">📅 Tenure</h4>
                <p style="font-size: 1.5rem; font-weight: 700; color: #333; margin: 0;">{tenure} years</p>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0 1rem 0;">
        <p style="font-size: 0.9rem;">Built with ❤️ using Streamlit & TensorFlow</p>
        <p style="font-size: 0.8rem; margin-top: 0.5rem;">© 2026 Customer Churn Prediction System</p>
    </div>
""", unsafe_allow_html=True)
