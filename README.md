# 📈 Stock Price Predictor
[![View App](https://img.shields.io/badge/Live%20App-Streamlit-blue?logo=streamlit)](https://hetron-stock-price-predictor.streamlit.app)

Predicts future stock prices using an ML pipeline that fetches live data, engineers technical features, selects the best model via cross-validation, and forecasts 10 business days ahead. Built with Streamlit.

---

## 🚀 Overview

Enter a ticker (e.g. `NVDA`) and the app:
- Fetches historical price data for the stock, SPY (market index), and VXX (volatility index) via the Tiingo API
- Engineers 11 technical features (RSI, price ratios, log returns, lag prices, moving averages)
- Compares Ridge, RandomForest, and GradientBoosting via **TimeSeriesSplit cross-validation**
- Evaluates the best model on a held-out **last 20% test set** (no data leakage)
- Displays forecast, evaluation metrics, model comparison, feature importance, and a walk-forward backtest

---

## 🌐 Live Demo

👉 [https://hetron-stock-price-predictor.streamlit.app](https://hetron-stock-price-predictor.streamlit.app)

---

## 📁 Project Structure

```
stock-price-predictor/
├── app.py                          # Minimal Streamlit entry point
├── src/
│   ├── config.py                   # All constants, feature lists, hyperparameters
│   ├── trading_predictor.py        # Core ML pipeline (fetch → features → train → predict)
│   ├── Hello.py                    # Welcome page
│   └── pages/
│       ├── 1_📊_StockAnalysis.py   # Fundamental analysis + news (yfinance)
│       └── 2_📈_StockPredictor.py  # Full prediction UI with metrics and charts
├── models/                         # Saved model artifacts
├── data/                           # CSV data (fallback / reference)
├── notebooks/                      # Research and EDA
├── tests/
│   ├── conftest.py                 # Shared synthetic data fixtures
│   ├── test_config.py              # Tests for config constants
│   ├── test_trading_predictor.py   # Tests for data prep and splitting
│   └── test_evaluation.py          # Tests for evaluation and model comparison
├── requirements.txt
└── .env                            # API keys (not committed)
```

---

## 🧠 ML Pipeline

### Feature Engineering (11 features)
| Feature | Description |
|---------|-------------|
| RSI | 14-day Relative Strength Index |
| price_ratio_to_index | Stock / SPY adjusted close |
| price_ratio_to_vxx | Stock / VXX adjusted close |
| price_diff_from_index | Stock − SPY |
| price_diff_from_vxx | Stock − VXX |
| log_returns | log(price[t] / price[t−1]) |
| volatility_adjusted_returns | log_returns / VXX price |
| lag_1_day | Adjusted close 1 day ago |
| lag_5_days | Adjusted close 5 days ago |
| lag_30_days | Adjusted close 30 days ago |
| lag_45_days | Adjusted close 45 days ago |

### Model Selection
Three candidates are compared using **5-fold TimeSeriesSplit CV** (test window always strictly after training window — no future data leakage):
- `Ridge(alpha=1.0)` — linear baseline
- `RandomForestRegressor` — non-linear, robust to outliers
- `GradientBoostingRegressor` — boosted trees, often best on tabular data

The model with the lowest mean CV RMSE is selected, re-trained on all data, and used for future predictions.

### Evaluation
The final model is evaluated on a held-out **last 20% of the chronological dataset** using:
- **RMSE** — Root Mean Squared Error (penalises large errors)
- **MAE** — Mean Absolute Error (average dollar error)
- **R²** — Coefficient of determination (1.0 = perfect)
- **MAPE** — Mean Absolute Percentage Error

### Walk-Forward Backtest
A separate backtesting chart shows what Ridge regression would have predicted on each historical test fold, training only on data available at the time of prediction.

---

## 🛠️ Setup

1. Clone the repo
2. Create a `.env` file:
   ```
   TIINGO_API_KEY=your_key_here
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
```

To see coverage:
```bash
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## ⚠️ Known Limitations

- **Recursive forecasting compounds errors** — each future prediction is used as input to the next; uncertainty grows with horizon
- **Lag features dominate** — the model learns "tomorrow ≈ today", which is a near-random-walk approximation
- **No macro regime detection** — the model doesn't know about interest rate changes, earnings events, or market crashes
- **API dependency** — requires a valid Tiingo API key; free tier has rate limits
