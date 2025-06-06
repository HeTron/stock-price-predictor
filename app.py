import streamlit as st
from src.trading_predictor import fetch_data, get_start_date, preprocess_data, training_data_prep, model_operation
from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv('TIINGO_API_KEY')

# Streamlit app
st.title('Stock Price Prediction')
st.write('Enter Stock Symbol')

stock_symbol = st.text_input('Stock Symbol').upper()

if st.button('Predict'):
    if stock_symbol:
        optimal_start_date = get_start_date(stock_symbol, token)

        # Fetch the historical data
        stock_data = fetch_data(stock_symbol, optimal_start_date, token)
        index_data = fetch_data('spy', optimal_start_date, token)
        vxx_data = fetch_data('vxx', optimal_start_date, token)

        stock_data, index_data, vxx_data = preprocess_data(stock_data, index_data, vxx_data)

        X_train, y_train = training_data_prep(stock_data)

        predictions_df = model_operation(X_train, y_train, stock_data)

        st.write(predictions_df)
    else:
        st.write("Please enter a valid stock symbol.")