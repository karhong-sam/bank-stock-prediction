# Stock Price Prediction App

## Overview
This is a **Stock Price Prediction App** built using **Streamlit** and **PyTorch**. The app allows users to fetch historical stock data, explore data through **Exploratory Data Analysis (EDA)**, train a **LSTM-RNN model**, evaluate its performance, and predict future stock prices.

## Features
- Fetch **historical stock data** from Yahoo Finance
- Perform **Exploratory Data Analysis (EDA)**, including **Moving Averages** and **MACD Histograms**
- Train a **LSTM-RNN model** to predict stock prices
- Evaluate model performance using **MAE, RMSE, R2 Score, and MAPE**
- Test trained model on unseen data
- Predict **future stock prices** based on trained model
- Download trained models for future use

## Installation
To run this app locally, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/karhong-sam/bank-stock-prediction.git
cd stock-price-prediction
```

### 2. Install Dependencies
Create a virtual environment (optional but recommended) and install required dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

## Usage
The app consists of **six main tabs**:

### 1️⃣ Disclaimer
- Provides important notices and legal disclaimers about stock predictions and investment risks.

### 2️⃣ Fetch Data
- Enter **Stock Ticker** (e.g., `1155.KL` for Maybank)
- Select **Start Date** and **End Date**
- Click `Fetch Data` to download stock data

### 3️⃣ View EDA (Exploratory Data Analysis)
- View stock data summary
- Plot **Moving Averages (MA20, MA60)**
- Visualize **MACD Histogram & Signal Line**

### 4️⃣ Train Model & Evaluate
- Preprocesses stock data for training
- Train an **LSTM-RNN** model
- Evaluate **Train & Validation Loss**
- Download trained model

### 5️⃣ Test Model Performance
- Evaluate the trained model on unseen test data
- Show **Actual vs Predicted Prices**
- Display **Error Metrics (MAE, RMSE, R2 Score)**

### 6️⃣ Predict Future Prices
- Upload a trained model (`.pkl` file) or use the latest trained model
- Select **Days to Predict**
- Plot **Future Price Predictions**

## Technologies Used
- **Streamlit** - For web-based visualization
- **yfinance** - Fetching stock market data
- **PyTorch** - LSTM-RNN model for time-series prediction
- **Plotly** - Interactive visualizations
- **scikit-learn** - Data preprocessing & evaluation metrics
- **Pandas & NumPy** - Data handling

## Requirements
All dependencies are listed in `requirements.txt`:
```
yfinance
numpy
pandas
matplotlib
seaborn
missingno
requests
selenium
wbdata
torch
torchinfo
scikit-learn
joblib
statsmodels
plotly
streamlit
Pillow
```

## Notes
- **CUDA Support**: If you have a **GPU**, PyTorch will use CUDA for faster training.
- **Data Handling**: The app automatically cleans and preprocesses stock data.
- **Performance**: LSTM-RNN models work best with **longer timeframes** and **relevant features**.

## License
This project is licensed under the **Apache License 2.0**.

## Acknowledgments
- **Yahoo Finance** for stock data
- **PyTorch Community** for deep learning support
- **Streamlit** for making data visualization easy

---
### 🚀 Happy Investing & Predicting! 📈
