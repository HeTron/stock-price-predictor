import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dotenv import load_dotenv

from src.trading_predictor import (
    fetch_data, get_start_date, preprocess_data,
    training_data_prep, model_operation, walk_forward_backtest,
)

load_dotenv()
token = os.getenv('TIINGO_API_KEY')

st.title('Stock Price Prediction')

stock_symbol = st.text_input('Enter Stock Symbol').upper()

if st.button('Predict'):
    if not stock_symbol:
        st.warning("Please enter a valid stock symbol.")
    else:
        with st.spinner(f'Fetching data and training models for {stock_symbol}...'):
            optimal_start_date = get_start_date(stock_symbol, token)

            stock_data = fetch_data(stock_symbol, optimal_start_date, token)
            index_data = fetch_data('spy', optimal_start_date, token)
            vxx_data = fetch_data('vxx', optimal_start_date, token)

            stock_data, index_data, vxx_data = preprocess_data(stock_data, index_data, vxx_data)

            X_train, y_train = training_data_prep(stock_data)

            predictions_df, metrics_dict, comparison_df, feature_importance_df = (
                model_operation(X_train, y_train, stock_data)
            )

            backtest_df = walk_forward_backtest(stock_data)

        # --- Prediction chart ---
        fig, ax = plt.subplots()
        ax.plot(predictions_df.index, predictions_df['Predicted Adj Close'], marker='o', label='Predicted Price')
        plt.xticks(rotation=45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.set_title(f'{stock_symbol} — 10-Day Price Forecast')
        ax.set_xlabel('Date')
        ax.set_ylabel('Predicted Adj Close')
        ax.legend()
        plt.tight_layout()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Forecast')
            st.dataframe(predictions_df.style.format({'Predicted Adj Close': '${:.2f}'}))
        with col2:
            st.subheader('Forecast Chart')
            st.pyplot(fig)

        st.divider()

        # --- Model evaluation metrics ---
        with st.expander('Model Evaluation Metrics', expanded=True):
            st.caption(
                f'Best model: **{comparison_df.iloc[0]["Model"]}** '
                f'(evaluated on held-out last 20% of data)'
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric('RMSE', f'${metrics_dict["RMSE"]:.2f}')
            m2.metric('MAE', f'${metrics_dict["MAE"]:.2f}')
            m3.metric('R²', f'{metrics_dict["R2"]:.4f}')
            m4.metric('MAPE', f'{metrics_dict["MAPE"]:.2f}%')

        # --- Model comparison table ---
        with st.expander('Model Comparison (Cross-Validation)'):
            st.caption('5-fold TimeSeriesSplit CV on training data. Lower RMSE = better.')
            display_df = comparison_df.copy()
            for col in ['RMSE_mean', 'RMSE_std', 'MAE_mean', 'R2_mean']:
                display_df[col] = display_df[col].round(4)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        # --- Feature importance ---
        with st.expander('Feature Importance'):
            fig_fi = px.bar(
                feature_importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title=f'Feature Importances — {comparison_df.iloc[0]["Model"]}',
            )
            fig_fi.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_fi, use_container_width=True)

        # --- Walk-forward backtest ---
        with st.expander('Historical Backtesting (Walk-Forward)'):
            st.caption(
                'Shows how well the model would have predicted historical prices '
                'using only data available at each point in time (no future leakage).'
            )
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(
                x=backtest_df['Date'], y=backtest_df['Actual'],
                mode='lines', name='Actual', line=dict(color='steelblue'),
            ))
            fig_bt.add_trace(go.Scatter(
                x=backtest_df['Date'], y=backtest_df['Predicted'],
                mode='lines', name='Predicted', line=dict(color='orange', dash='dash'),
            ))
            fig_bt.update_layout(
                title='Actual vs Predicted — Walk-Forward Backtest',
                xaxis_title='Date', yaxis_title='Adj Close',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            )
            st.plotly_chart(fig_bt, use_container_width=True)

            backtest_rmse = float(
                ((backtest_df['Actual'] - backtest_df['Predicted']) ** 2).mean() ** 0.5
            )
            st.metric('Backtest RMSE', f'${backtest_rmse:.2f}')
