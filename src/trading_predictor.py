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
from sklearn.model_selection import TimeSeriesSplit

try:
    from src.config import (
        FEATURE_COLUMNS, TARGET_COLUMN, TEST_SIZE_FRACTION, MIN_TRAIN_ROWS,
        TSCV_N_SPLITS, FORECAST_DAYS, LAG_COL_INDICES, RSI_WINDOW, MA_WINDOWS,
        MODEL_RIDGE_ALPHA, MODEL_RF_N_ESTIMATORS, MODEL_RF_MAX_DEPTH, MODEL_RF_N_JOBS,
        MODEL_GB_N_ESTIMATORS, MODEL_GB_MAX_DEPTH, MODEL_GB_LEARNING_RATE, LOG_FORMAT,
    )
except ImportError:
    from config import (
        FEATURE_COLUMNS, TARGET_COLUMN, TEST_SIZE_FRACTION, MIN_TRAIN_ROWS,
        TSCV_N_SPLITS, FORECAST_DAYS, LAG_COL_INDICES, RSI_WINDOW, MA_WINDOWS,
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
        logger.warning("startDate not found in response for %s. Defaulting to 2019-01-01. Response: %s",
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
        logger.error("Unexpected JSON response format for %s: %s", stock_symbol, json_response)
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

    for lag in [1, 5, 30, 45]:
        stock_data[f'lag_{lag}_day' if lag == 1 else f'lag_{lag}_days'] = (
            stock_data[TARGET_COLUMN].shift(lag)
        )

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
    X = stock_data[FEATURE_COLUMNS].values
    y = stock_data[TARGET_COLUMN].values
    return X, y


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _mape(y_true, y_pred):
    mask = y_true != 0
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_model(model, X_test, y_test):
    """Returns RMSE, MAE, R², and MAPE on held-out test data."""
    y_pred = model.predict(X_test)
    metrics = {
        'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'MAE': float(mean_absolute_error(y_test, y_pred)),
        'R2': float(r2_score(y_test, y_pred)),
        'MAPE': _mape(y_test, y_pred),
    }
    logger.info("Test metrics — RMSE: %.4f | MAE: %.4f | R²: %.4f | MAPE: %.2f%%",
                metrics['RMSE'], metrics['MAE'], metrics['R2'], metrics['MAPE'])
    return metrics


def cross_validate_model(pipeline, X, y, n_splits=TSCV_N_SPLITS):
    """TimeSeriesSplit CV — test window is always strictly after training window."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = {'RMSE': [], 'MAE': [], 'R2': [], 'MAPE': []}

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        import sklearn.base
        fold_model = sklearn.base.clone(pipeline)
        fold_model.fit(X_tr, y_tr)

        fold_m = evaluate_model(fold_model, X_te, y_te)
        for k in fold_metrics:
            fold_metrics[k].append(fold_m[k])
        logger.info("Fold %d — RMSE: %.4f", fold, fold_m['RMSE'])

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
    Returns (best_fitted_pipeline, comparison_df) where the pipeline is
    re-fit on the full X_train/y_train.
    """
    candidates = _build_candidate_models()
    rows = []

    for name, pipeline in candidates.items():
        logger.info("Cross-validating %s...", name)
        fold_metrics = cross_validate_model(pipeline, X_train, y_train)
        rows.append({
            'Model': name,
            'RMSE_mean': float(np.mean(fold_metrics['RMSE'])),
            'RMSE_std': float(np.std(fold_metrics['RMSE'])),
            'MAE_mean': float(np.mean(fold_metrics['MAE'])),
            'R2_mean': float(np.mean(fold_metrics['R2'])),
        })

    comparison_df = pd.DataFrame(rows).sort_values('RMSE_mean').reset_index(drop=True)
    best_name = comparison_df.iloc[0]['Model']
    logger.info("Best model: %s (CV RMSE: %.4f)", best_name, comparison_df.iloc[0]['RMSE_mean'])

    best_pipeline = _build_candidate_models()[best_name]
    best_pipeline.fit(X_train, y_train)
    return best_pipeline, comparison_df


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def get_feature_importance(fitted_pipeline, feature_names=None):
    """
    Extracts feature importances from the pipeline's final estimator.
    Ridge → abs(coef_). RandomForest / GradientBoosting → feature_importances_.
    Returns a DataFrame sorted descending by Importance.
    """
    if feature_names is None:
        feature_names = FEATURE_COLUMNS

    estimator = fitted_pipeline.steps[-1][1]

    if hasattr(estimator, 'feature_importances_'):
        importances = estimator.feature_importances_
    elif hasattr(estimator, 'coef_'):
        importances = np.abs(estimator.coef_).flatten()
    else:
        logger.warning("Cannot extract feature importances from %s", type(estimator).__name__)
        return pd.DataFrame({'Feature': feature_names, 'Importance': [np.nan] * len(feature_names)})

    df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    return df.sort_values('Importance', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Walk-forward backtesting
# ---------------------------------------------------------------------------

def walk_forward_backtest(stock_data, n_splits=TSCV_N_SPLITS):
    """
    Expanding-window walk-forward validation.
    Each fold trains only on data before the test window — no future data leakage.
    Returns a DataFrame with Date, Actual, Predicted, AbsError.
    """
    X = stock_data[FEATURE_COLUMNS].values
    y = stock_data[TARGET_COLUMN].values
    dates = np.array(stock_data['date'].values)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        pipeline = make_pipeline(StandardScaler(), Ridge(alpha=MODEL_RIDGE_ALPHA))
        pipeline.fit(X[train_idx], y[train_idx])
        y_pred = pipeline.predict(X[test_idx])

        for date, actual, predicted in zip(dates[test_idx], y[test_idx], y_pred):
            results.append({
                'Date': date,
                'Actual': float(actual),
                'Predicted': float(predicted),
                'AbsError': float(abs(actual - predicted)),
            })
        logger.info("Backtest fold %d — %d test points", fold, len(test_idx))

    return pd.DataFrame(results).sort_values('Date').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main model operation
# ---------------------------------------------------------------------------

def model_operation(X_train, y_train, stock_data):
    """
    Full model pipeline: compare models, evaluate on hold-out, generate future predictions.

    Returns:
        predictions_df     — DataFrame with 10 future business day price predictions
        metrics_dict       — dict with RMSE, MAE, R², MAPE on held-out test set
        comparison_df      — DataFrame comparing all candidate models via CV
        feature_importance_df — DataFrame of feature importances from best model
    """
    n = len(X_train)
    if n < MIN_TRAIN_ROWS:
        logger.error("Not enough data: %d rows (minimum %d)", n, MIN_TRAIN_ROWS)
        raise ValueError(f"Need at least {MIN_TRAIN_ROWS} rows to train; got {n}.")

    # Chronological split — use training portion for model selection
    split_idx = int(n * (1 - TEST_SIZE_FRACTION))
    X_tr, X_te = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_te = y_train[:split_idx], y_train[split_idx:]
    logger.info("Model selection on %d rows; evaluation on %d rows", len(X_tr), len(X_te))

    # Compare models on training portion (no peeking at test set)
    _, comparison_df = compare_models(X_tr, y_tr)

    # Rebuild and train the best model type on ALL data for final predictions
    best_name = comparison_df.iloc[0]['Model']
    final_model = _build_candidate_models()[best_name]
    final_model.fit(X_train, y_train)
    logger.info("Final model (%s) trained on full dataset (%d rows)", best_name, n)

    # Evaluate using a model trained only on the training split (no leakage)
    eval_model = _build_candidate_models()[best_name]
    eval_model.fit(X_tr, y_tr)
    metrics_dict = evaluate_model(eval_model, X_te, y_te)

    # Feature importance from the full-data model
    feature_importance_df = get_feature_importance(final_model)

    # Recursive future prediction
    last_row = stock_data.iloc[-1]
    features = last_row[FEATURE_COLUMNS].values.reshape(1, -1).astype(float)

    predictions = []
    last_date = pd.to_datetime(stock_data['date'].max())
    future_dates = pd.date_range(last_date, periods=FORECAST_DAYS + 1, freq='B')[1:]

    idx_lag1 = LAG_COL_INDICES['lag_1_day']
    idx_lag5 = LAG_COL_INDICES['lag_5_days']
    idx_lag30 = LAG_COL_INDICES['lag_30_days']
    idx_lag45 = LAG_COL_INDICES['lag_45_days']

    for _ in range(FORECAST_DAYS):
        next_price = float(final_model.predict(features)[0])
        predictions.append(next_price)
        # Shift lags in reverse order to avoid overwriting before reading
        features[0][idx_lag45] = features[0][idx_lag30]
        features[0][idx_lag30] = features[0][idx_lag5]
        features[0][idx_lag5] = features[0][idx_lag1]
        features[0][idx_lag1] = next_price

    predictions_df = pd.DataFrame({
        'Date': future_dates.date,
        'Predicted Adj Close': predictions,
    }).set_index('Date')

    return predictions_df, metrics_dict, comparison_df, feature_importance_df
