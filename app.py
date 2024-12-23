import streamlit as st
from PIL import Image
from utils import LSTMRNN, fetch_and_process_data, view_eda, train_and_evaluate, test_model_performance, predict_future_prices

# Streamlit App with Tabs
st.title("Stock Price Prediction App")

# Display an introductory image
intro_image = Image.open("images/gradient-stock-market-concept-with-statistics.png")
st.image(intro_image, use_container_width=True)

# Tabs
tabs = st.tabs(["Fetch Data", "View EDA", "Train Model & Evaluate", "Test Model Performance", "Predict Future Prices"])

# Tab 1: Fetch Data
with tabs[0]:
    st.header("Step 1: Fetch Data")
    stock_ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", "AAPL")
    start_date = st.date_input("Select Start Date")
    end_date = st.date_input("Select End Date")

    if st.button("Fetch Data"):
        fetch_and_process_data(stock_ticker, start_date, end_date)

# Tab 2: View EDA
with tabs[1]:
    st.header("Step 2: View EDA")
    view_eda()

# Tab 3: Train Model & Evaluate
with tabs[2]:
    st.header("Step 3: Train Model & Evaluate")
    if 'data' in st.session_state:
        train_and_evaluate()
    else:
        st.error("Please fetch data first.")

# Tab 4: Test Model Performance
with tabs[3]:
    st.header("Step 4: Test Model Performance")
    if 'model' in st.session_state:
        test_model_performance()
    else:
        st.error("Please train the model first.")

# Tab 5: Predict Future Prices
with tabs[4]:
    st.header("Step 5: Predict Future Prices")
    predict_future_prices()
