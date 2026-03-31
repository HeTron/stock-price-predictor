"""Shared fixtures for all test modules."""
import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta


def _make_stock_df(n=200, seed=42):
    """
    Build a synthetic stock DataFrame that mimics the output of preprocess_data().
    Prices follow a simple random walk so lag/RSI features are non-trivial.
    """
    rng = np.random.default_rng(seed)
    prices = 100.0 + np.cumsum(rng.normal(0, 1, n))
    prices = np.clip(prices, 10, None)

    start = date(2022, 1, 4)
    dates = [start + timedelta(days=i) for i in range(n)]

    df = pd.DataFrame({
        'date': dates,
        'adjClose': prices,
        'close': prices * 1.001,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'open': prices * 0.999,
        'volume': rng.integers(1_000_000, 10_000_000, n).astype(float),
    })

    index_prices = 400.0 + np.cumsum(rng.normal(0, 0.5, n))
    index_prices = np.clip(index_prices, 50, None)
    vxx_prices = 20.0 + np.cumsum(rng.normal(0, 0.1, n))
    vxx_prices = np.clip(vxx_prices, 5, None)

    # RSI
    delta = pd.Series(prices).diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = (100 - (100 / (1 + rs))).fillna(50)

    # Ratio / diff features
    df['price_ratio_to_index'] = prices / index_prices
    df['price_ratio_to_vxx']   = prices / vxx_prices
    df['price_diff_from_index'] = prices - index_prices
    df['price_diff_from_vxx']   = prices - vxx_prices

    # Log returns
    log_ret = np.log(pd.Series(prices) / pd.Series(prices).shift(1)).fillna(0)
    df['log_returns'] = log_ret.values
    df['volatility_adjusted_returns'] = log_ret.values / vxx_prices

    # Moving averages
    for w in [14, 30, 90]:
        df[f'ma_{w}']          = pd.Series(prices).rolling(w).mean().fillna(prices[0])
        df[f'index_ma_{w}']    = pd.Series(index_prices).rolling(w).mean().fillna(index_prices[0])
        df[f'stock_over_ma_{w}']  = prices / df[f'ma_{w}']
        df[f'index_over_ma_{w}'] = index_prices / df[f'index_ma_{w}']

    # Lag RETURN features (replacing absolute price lags)
    df['lag_return_1']  = log_ret.shift(1).fillna(0).values
    df['lag_return_5']  = log_ret.shift(5).fillna(0).values
    df['lag_return_30'] = log_ret.shift(30).fillna(0).values
    df['lag_return_45'] = log_ret.shift(45).fillna(0).values

    return df


@pytest.fixture
def sample_stock_df():
    return _make_stock_df(n=200)


@pytest.fixture
def small_stock_df():
    return _make_stock_df(n=80)
