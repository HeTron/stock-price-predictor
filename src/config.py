import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# --- Feature Engineering ---
RSI_WINDOW = 14
MA_WINDOWS = [14, 30, 90]
LAG_DAYS = [1, 5, 30, 45]

# --- Feature Columns used for training (order matters — matches array indices) ---
FEATURE_COLUMNS = [
    'RSI',
    'price_ratio_to_index',
    'price_ratio_to_vxx',
    'price_diff_from_index',
    'price_diff_from_vxx',
    'log_returns',
    'volatility_adjusted_returns',
    'lag_1_day',
    'lag_5_days',
    'lag_30_days',
    'lag_45_days',
]

TARGET_COLUMN = 'adjClose'

# Indices of lag columns within FEATURE_COLUMNS — used for recursive future prediction
LAG_COL_INDICES = {
    'lag_1_day': FEATURE_COLUMNS.index('lag_1_day'),
    'lag_5_days': FEATURE_COLUMNS.index('lag_5_days'),
    'lag_30_days': FEATURE_COLUMNS.index('lag_30_days'),
    'lag_45_days': FEATURE_COLUMNS.index('lag_45_days'),
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
