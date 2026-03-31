import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# --- Feature Engineering ---
RSI_WINDOW = 14
MA_WINDOWS = [14, 30, 90]

# --- Feature Columns used for training (order matters — matches array indices) ---
# Lag features use log RETURNS (not absolute prices) so the model is scale-invariant.
# Absolute price lags caused predictions to anchor to historical price regimes
# (e.g. split-adjusted AAPL 2015 adjClose ~$25 vs current ~$180).
FEATURE_COLUMNS = [
    'RSI',
    'price_ratio_to_index',
    'price_ratio_to_vxx',
    'price_diff_from_index',
    'price_diff_from_vxx',
    'log_returns',
    'volatility_adjusted_returns',
    'lag_return_1',    # log return 1 day ago
    'lag_return_5',    # log return 5 days ago
    'lag_return_30',   # log return 30 days ago
    'lag_return_45',   # log return 45 days ago
]

TARGET_COLUMN = 'adjClose'   # price column used throughout preprocessing
MODEL_TARGET = 'log_returns' # what the model actually predicts; converted back to prices for output

# Indices of lag return columns — used for recursive future prediction
LAG_RETURN_COL_INDICES = {
    'lag_return_1':  FEATURE_COLUMNS.index('lag_return_1'),
    'lag_return_5':  FEATURE_COLUMNS.index('lag_return_5'),
    'lag_return_30': FEATURE_COLUMNS.index('lag_return_30'),
    'lag_return_45': FEATURE_COLUMNS.index('lag_return_45'),
}

# --- Train / Test Split ---
TEST_SIZE_FRACTION = 0.2       # last 20% of chronological data held out
MIN_TRAIN_ROWS = 200           # refuse to train if fewer rows after split

# --- Cross-Validation ---
TSCV_N_SPLITS = 5              # TimeSeriesSplit folds

# --- Model Hyperparameters ---
MODEL_RIDGE_ALPHA = 1.0

MODEL_RF_N_ESTIMATORS = 50
MODEL_RF_MAX_DEPTH = 10
MODEL_RF_N_JOBS = -1           # use all CPU cores

MODEL_GB_N_ESTIMATORS = 50
MODEL_GB_MAX_DEPTH = 4
MODEL_GB_LEARNING_RATE = 0.1

# --- Forecast ---
FORECAST_DAYS = 10             # number of future business days to predict

# --- Logging ---
LOG_FORMAT = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
