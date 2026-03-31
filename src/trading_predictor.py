import logging
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

try:
    from src.config import (
        FEATURE_COLUMNS, TARGET_COLUMN, MODEL_TARGET,
        TEST_SIZE_FRACTION, MIN_TRAIN_ROWS, TSCV_N_SPLITS, FORECAST_DAYS,
        LAG_RETURN_COL_INDICES, RSI_WINDOW, MA_WINDOWS,
        MODEL_RIDGE_ALPHA, MODEL_RF_N_ESTIMATORS, MODEL_RF_MAX_DEPTH, MODEL_RF_N_JOBS,
        MODEL_GB_N_ESTIMATORS, MODEL_GB_MAX_DEPTH, MODEL_GB_LEARNING_RATE, LOG_FORMAT,
    )
except ImportError:
    from config import (
        FEATURE_COLUMNS, TARGET_COLUMN, MODEL_TARGET,
        TEST_SIZE_FRACTION, MIN_TRAIN_ROWS, TSCV_N_SPLITS, FORECAST_DAYS,
        LAG_RETURN_COL_INDICES, RSI_WINDOW, MA_WINDOWS,
        MODEL_RIDGE_ALPHA, MODEL_RF_N_ESTIMATORS, MODEL_RF_MAX_DEPTH, MODEL_RF_N_JOBS,
        MODEL_GB_N_ESTIMATORS, MODEL_GB_MAX_DEPTH, MODEL_GB_LEARNING_RATE, LOG_FORMAT,
    )

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def get_start_date(stock_symbol, token, max_years=15):
    headers = {'Content-Type': 'application/json'}
    meta_url = f'https://api.tiingo.com/tiingo/daily/{stock_symbol}?token={token}'
    meta_response = requests.get(meta_url, headers=headers)
    meta_data = meta_response.json()

    if 'startDate' not in meta_data:
        logger.warning("startDate not found for %s. Defaulting to 2019-01-01. Response: %s",
                       stock_symbol, meta_data)
        return "2019-01-01"

    start_date = datetime.strptime(meta_data['startDate'], '%Y-%m-%d')
    years_ago_date = datetime.now() - timedelta(days=max_years * 365)
    optimal_start_date = max(start_date, years_ago_date).strftime('%Y-%m-%d')
    logger.info("Optimal start date for %s: %s", stock_symbol, optimal_start_date)
    return optimal_start_date


def fetch_data(stock_symbol, optimal_start_date, token):
    headers = {'Content-Type': 'application/json'}
    url = (f'https://api.tiingo.com/tiingo/daily/{stock_symbol}/prices'
           f'?startDate={optimal_start_date}&token={token}')
    response = requests.get(url, headers=headers)
    json_response = response.json()

    if isinstance(json_response, list):
        df = pd.DataFrame(json_response)
    elif isinstance(json_response, dict):
        df = pd.DataFrame([json_response])
    else:
        logger.error("Unexpected JSON response for %s: %s", stock_symbol, json_response)
        return pd.DataFrame()

    logger.info("Fetched %d rows for %s", len(df), stock_symbol)
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def calculate_rsi(data, window=RSI_WINDOW):
    delta = data[TARGET_COLUMN].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def preprocess_data(stock_data, index_data, vxx_data):
    for df in (stock_data, index_data, vxx_data):
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
        df.drop(['divCash', 'splitFactor'], axis=1, inplace=True)

    stock_data['RSI'] = calculate_rsi(stock_data).fillna(50)

    stock_data['price_ratio_to_index'] = stock_data[TARGET_COLUMN] / index_data[TARGET_COLUMN]
    stock_data['price_ratio_to_vxx'] = stock_data[TARGET_COLUMN] / vxx_data[TARGET_COLUMN]
    stock_data['price_diff_from_index'] = stock_data[TARGET_COLUMN] - index_data[TARGET_COLUMN]
    stock_data['price_diff_from_vxx'] = stock_data[TARGET_COLUMN] - vxx_data[TARGET_COLUMN]

    stock_data['log_returns'] = np.log(
        stock_data[TARGET_COLUMN] / stock_data[TARGET_COLUMN].shift(1)
    )
    stock_data['volatility_adjusted_returns'] = (
        stock_data['log_returns'] / vxx_data[TARGET_COLUMN]
    )

    for window in MA_WINDOWS:
        stock_data[f'ma_{window}'] = stock_data[TARGET_COLUMN].rolling(window=window).mean()
        index_ma = index_data[TARGET_COLUMN].rolling(window=window).mean()
        stock_data[f'index_ma_{window}'] = index_ma
        stock_data[f'stock_over_ma_{window}'] = stock_data[TARGET_COLUMN] / stock_data[f'ma_{window}']
        stock_data[f'index_over_ma_{window}'] = index_data[TARGET_COLUMN] / index_ma

    # Lag features use log RETURNS, not absolute prices.
    # This keeps features scale-invariant so the model generalises across
    # different price regimes (e.g. pre/post stock splits).
    log_ret = stock_data['log_returns']
    stock_data['lag_return_1']  = log_ret.shift(1)
    stock_data['lag_return_5']  = log_ret.shift(5)
    stock_data['lag_return_30'] = log_ret.shift(30)
    stock_data['lag_return_45'] = log_ret.shift(45)

    stock_data = stock_data.dropna().reset_index(drop=True)
    logger.info("Preprocessed data: %d rows remaining after dropna", len(stock_data))
    return stock_data, index_data, vxx_data


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def split_time_series(stock_data, test_size=TEST_SIZE_FRACTION):
    """Chronological train/test split — never shuffles."""
    n = len(stock_data)
    split_idx = int(n * (1 - test_size))
    train_df = stock_data.iloc[:split_idx].copy()
    test_df = stock_data.iloc[split_idx:].copy()
    logger.info("Train/test split: %d train rows, %d test rows", len(train_df), len(test_df))
    return train_df, test_df


# ---------------------------------------------------------------------------
# Training data preparation
# ---------------------------------------------------------------------------

def training_data_prep(stock_data):
    """Returns X (features) and y (log returns — the model target)."""
    X = stock_data[FEATURE_COLUMNS].values
    y = stock_data[MODEL_TARGET].values   # log_returns, not adjClose
    return X, y


# ---------------------------------------------------------------------------
# Evaluation (metrics reported in price space for interpretability)
# ---------------------------------------------------------------------------

def _mape(y_true, y_pred):
    mask = y_true != 0
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_model(model, X_test, y_test):
    """
    Computes RMSE, MAE, R², MAPE directly on whatever y_test contains.
    In the main pipeline these are price-space values for display clarity.
    """
    y_pred = model.predict(X_test)
    metrics = {
        'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'MAE': float(mean_absolute_error(y_test, y_pred)),
        'R2': float(r2_score(y_test, y_pred)),
        'MAPE': _mape(y_test, y_pred),
    }
    logger.info("Metrics — RMSE: %.4f | MAE: %.4f | R²: %.4f | MAPE: %.2f%%",
                metrics['RMSE'], metrics['MAE'], metrics['R2'], metrics['MAPE'])
    return metrics


def _return_preds_to_price_metrics(pred_log_returns, actual_prices, prev_prices):
    """Convert predicted log returns → price predictions, then compute price-space metrics."""
    pred_prices = prev_prices * np.exp(pred_log_returns)
    return {
        'RMSE': float(np.sqrt(mean_squared_error(actual_prices, pred_prices))),
        'MAE':  float(mean_absolute_error(actual_prices, pred_prices)),
        'R2':   float(r2_score(actual_prices, pred_prices)),
        'MAPE': _mape(actual_prices, pred_prices),
    }


def cross_validate_model(pipeline, X, y, n_splits=TSCV_N_SPLITS):
    """TimeSeriesSplit CV on log-return target. Metrics are in return space (used for model selection)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = {'RMSE': [], 'MAE': [], 'R2': [], 'MAPE': []}

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        fold_model = clone(pipeline)
        fold_model.fit(X[train_idx], y[train_idx])
        fold_m = evaluate_model(fold_model, X[test_idx], y[test_idx])
        for k in fold_metrics:
            fold_metrics[k].append(fold_m[k])
        logger.info("Fold %d — RMSE: %.6f", fold, fold_m['RMSE'])

    return fold_metrics


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def _build_candidate_models():
    return {
        'Ridge': make_pipeline(
            StandardScaler(),
            Ridge(alpha=MODEL_RIDGE_ALPHA)
        ),
        'RandomForest': make_pipeline(
            StandardScaler(),
            RandomForestRegressor(
                n_estimators=MODEL_RF_N_ESTIMATORS,
                max_depth=MODEL_RF_MAX_DEPTH,
                n_jobs=MODEL_RF_N_JOBS,
                random_state=42,
            )
        ),
        'GradientBoosting': make_pipeline(
            StandardScaler(),
            GradientBoostingRegressor(
                n_estimators=MODEL_GB_N_ESTIMATORS,
                max_depth=MODEL_GB_MAX_DEPTH,
                learning_rate=MODEL_GB_LEARNING_RATE,
                random_state=42,
            )
        ),
    }


def compare_models(X_train, y_train):
    """
    Compares Ridge, RandomForest, GradientBoosting via TimeSeriesSplit CV.
    Returns (best_fitted_pipeline, comparison_df).
    """
    candidates = _build_candidate_models()
    rows = []

    for name, pipeline in candidates.items():
        logger.info("Cross-validating %s...", name)
        fold_metrics = cross_validate_model(pipeline, X_train, y_train)
        rows.append({
            'Model': name,
            'RMSE_mean': float(np.mean(fold_metrics['RMSE'])),
            'RMSE_std':  float(np.std(fold_metrics['RMSE'])),
            'MAE_mean':  float(np.mean(fold_metrics['MAE'])),
            'R2_mean':   float(np.mean(fold_metrics['R2'])),
        })

    comparison_df = pd.DataFrame(rows).sort_values('RMSE_mean').reset_index(drop=True)
    best_name = comparison_df.iloc[0]['Model']
    logger.info("Best model: %s (CV RMSE: %.6f)", best_name, comparison_df.iloc[0]['RMSE_mean'])

    best_pipeline = _build_candidate_models()[best_name]
    best_pipeline.fit(X_train, y_train)
    return best_pipeline, comparison_df


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def get_feature_importance(fitted_pipeline, feature_names=None):
    if feature_names is None:
        feature_names = FEATURE_COLUMNS

    estimator = fitted_pipeline.steps[-1][1]

    if hasattr(estimator, 'feature_importances_'):
        importances = estimator.feature_importances_
    elif hasattr(estimator, 'coef_'):
        importances = np.abs(estimator.coef_).flatten()
    else:
        logger.warning("Cannot extract importances from %s", type(estimator).__name__)
        return pd.DataFrame({'Feature': feature_names, 'Importance': [np.nan] * len(feature_names)})

    df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    return df.sort_values('Importance', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Walk-forward backtesting
# ---------------------------------------------------------------------------

def walk_forward_backtest(stock_data, n_splits=TSCV_N_SPLITS):
    """
    Expanding-window walk-forward validation.
    Model predicts log returns; results are converted to price space for display.
    """
    X      = stock_data[FEATURE_COLUMNS].values
    y      = stock_data[MODEL_TARGET].values      # log_returns
    prices = stock_data[TARGET_COLUMN].values     # adjClose for price conversion
    dates  = np.array(stock_data['date'].values)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        pipeline = make_pipeline(StandardScaler(), Ridge(alpha=MODEL_RIDGE_ALPHA))
        pipeline.fit(X[train_idx], y[train_idx])
        y_pred_returns = pipeline.predict(X[test_idx])

        for test_i, pred_return in zip(test_idx, y_pred_returns):
            actual_price = float(prices[test_i])
            prev_price   = float(prices[test_i - 1]) if test_i > 0 else actual_price
            pred_price   = prev_price * np.exp(pred_return)
            results.append({
                'Date':      dates[test_i],
                'Actual':    actual_price,
                'Predicted': pred_price,
                'AbsError':  abs(actual_price - pred_price),
            })
        logger.info("Backtest fold %d — %d test points", fold, len(test_idx))

    return pd.DataFrame(results).sort_values('Date').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main model operation
# ---------------------------------------------------------------------------

def model_operation(X_train, y_train, stock_data):
    """
    Full model pipeline: compare models, evaluate on hold-out, generate future predictions.

    X_train / y_train : features and log-return targets from training_data_prep()
    stock_data        : preprocessed DataFrame (needed for price context)

    Returns:
        predictions_df        — 10 future business days with predicted prices
        metrics_dict          — RMSE, MAE, R², MAPE in price space on held-out test set
        comparison_df         — CV comparison of all candidate models
        feature_importance_df — feature importances from best model
    """
    n = len(X_train)
    if n < MIN_TRAIN_ROWS:
        raise ValueError(f"Need at least {MIN_TRAIN_ROWS} rows to train; got {n}.")

    prices = stock_data[TARGET_COLUMN].values  # adjClose, for price-space metrics

    # Chronological 80/20 split
    split_idx = int(n * (1 - TEST_SIZE_FRACTION))
    X_tr, X_te = X_train[:split_idx], X_train[split_idx:]
    y_tr        = y_train[:split_idx]           # log returns (training)
    logger.info("Model selection on %d rows; evaluation on %d rows", split_idx, n - split_idx)

    # Compare models on training portion only (no test leakage)
    _, comparison_df = compare_models(X_tr, y_tr)
    best_name = comparison_df.iloc[0]['Model']

    # Rebuild best model and train on ALL data for final predictions
    final_model = _build_candidate_models()[best_name]
    final_model.fit(X_train, y_train)
    logger.info("Final model (%s) trained on full dataset (%d rows)", best_name, n)

    # Evaluate on held-out test set — convert return predictions to price space
    eval_model = _build_candidate_models()[best_name]
    eval_model.fit(X_tr, y_tr)
    pred_log_returns = eval_model.predict(X_te)
    actual_prices = prices[split_idx:]
    prev_prices   = prices[split_idx - 1 : n - 1]
    metrics_dict  = _return_preds_to_price_metrics(pred_log_returns, actual_prices, prev_prices)

    # Feature importance
    feature_importance_df = get_feature_importance(final_model)

    # Recursive future prediction: predict log return → convert to price
    last_row = stock_data.iloc[-1]
    features = last_row[FEATURE_COLUMNS].values.reshape(1, -1).astype(float)
    current_price = float(stock_data[TARGET_COLUMN].iloc[-1])

    predictions = []
    last_date = pd.to_datetime(stock_data['date'].max())
    future_dates = pd.date_range(last_date, periods=FORECAST_DAYS + 1, freq='B')[1:]

    idx1  = LAG_RETURN_COL_INDICES['lag_return_1']
    idx5  = LAG_RETURN_COL_INDICES['lag_return_5']
    idx30 = LAG_RETURN_COL_INDICES['lag_return_30']
    idx45 = LAG_RETURN_COL_INDICES['lag_return_45']

    for _ in range(FORECAST_DAYS):
        pred_return   = float(final_model.predict(features)[0])
        current_price = current_price * np.exp(pred_return)
        predictions.append(current_price)
        # Shift lag return features in reverse order
        features[0][idx45] = features[0][idx30]
        features[0][idx30] = features[0][idx5]
        features[0][idx5]  = features[0][idx1]
        features[0][idx1]  = pred_return

    predictions_df = pd.DataFrame({
        'Date': future_dates.date,
        'Predicted Adj Close': predictions,
    }).set_index('Date')

    return predictions_df, metrics_dict, comparison_df, feature_importance_df
