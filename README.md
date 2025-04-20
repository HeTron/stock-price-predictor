# 📈 Stock Price Predictor
[![View App](https://img.shields.io/badge/Live%20App-Streamlit-blue?logo=streamlit)](https://hetron-stock-price-predictor.streamlit.app)

This project predicts future stock prices using a machine learning model trained on historical price data and macro indicators. Built using Streamlit, it lets users enter a stock symbol and view predictions in real-time.

---

## 🚀 Overview

Users enter a stock ticker (e.g. NVDA), and the app:
- Automatically fetches historical stock/index/volatility data
- Preprocesses and aligns the data
- Applies a trained model to predict future values
- Visualizes the results with interactive charts

---

## 🌐 Live Demo

Try it here on Streamlit Cloud:  
👉 [https://hetron-stock-price-predictor.streamlit.app](https://hetron-stock-price-predictor.streamlit.app)

---

## 📁 Project Structure
```
stock-price-predictor/ 
├── app.py # Streamlit app 
├── src/ # Core logic and data handling 
├── models/ # Saved model files 
├── data/ # Raw CSVs (optional, fallback if API fails) 
├── notebooks/ # Research/EDA notebooks 
├── .env.example # Environment template (for API keys) 
├── requirements.txt 
└── README.md
```
---

## 🛠️ How to Run

1. Clone the repo  
2. Create a `.env` file using the `.env.example` template  
3. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:
    ```bash
    streamlit run app.py
    ```

## 💡 Notes

This was a capstone project for my data science bootcamp. It demonstrates real-world model deployment using API calls, data preprocessing, and Streamlit for delivery.