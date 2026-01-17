# Customer Churn Prediction App

A Streamlit web application for predicting customer churn using an Artificial Neural Network (ANN) model.

## Features

- Interactive web interface for inputting customer data
- Real-time churn prediction
- Pre-trained ANN model for accurate predictions
- Built with Streamlit and TensorFlow

## Files Included

- `app.py` - Main Streamlit application
- `model.h5` - Trained ANN model
- `label_encoder_gender.pkl` - Gender label encoder
- `onehot_encoder_geo.pkl` - Geography one-hot encoder
- `scaler.pkl` - Feature scaler
- `requirements.txt` - Python dependencies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/ann-churn-prediction.git
cd ann-churn-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Input Features

- **Geography**: Customer's geographical location
- **Gender**: Customer's gender
- **Age**: Customer's age (18-92)
- **Balance**: Account balance
- **Credit Score**: Customer's credit score
- **Estimated Salary**: Customer's estimated salary
- **Tenure**: Years with the bank (0-10)
- **Number of Products**: Number of products held (1-4)
- **Has Credit Card**: Whether customer has a credit card (0/1)
- **Is Active Member**: Whether customer is an active member (0/1)

## Output

The app provides:
- Churn probability (0-1)
- Prediction: Whether the customer is likely to churn

## Technologies Used

- Python 3.x
- Streamlit
- TensorFlow/Keras
- Scikit-learn
- Pandas
- NumPy

## License

MIT License
