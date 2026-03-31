"""Tests for src/config.py."""
import os
import pytest
from src.config import (
    FEATURE_COLUMNS, TARGET_COLUMN, MODEL_TARGET,
    TEST_SIZE_FRACTION, MIN_TRAIN_ROWS, TSCV_N_SPLITS, FORECAST_DAYS,
    LAG_RETURN_COL_INDICES, MA_WINDOWS, MODELS_DIR, DATA_DIR, BASE_DIR,
)


def test_feature_columns_count():
    assert len(FEATURE_COLUMNS) == 11


def test_feature_columns_are_strings():
    assert all(isinstance(c, str) for c in FEATURE_COLUMNS)


def test_lag_return_columns_all_in_feature_columns():
    for col in ['lag_return_1', 'lag_return_5', 'lag_return_30', 'lag_return_45']:
        assert col in FEATURE_COLUMNS, f"{col} missing from FEATURE_COLUMNS"


def test_no_absolute_price_lag_columns():
    """Absolute price lags cause regime-shift errors — they must not be in FEATURE_COLUMNS."""
    for col in ['lag_1_day', 'lag_5_days', 'lag_30_days', 'lag_45_days']:
        assert col not in FEATURE_COLUMNS, f"{col} should not be in FEATURE_COLUMNS (use lag_return_* instead)"


def test_lag_return_col_indices_match_feature_columns():
    for col, idx in LAG_RETURN_COL_INDICES.items():
        assert FEATURE_COLUMNS[idx] == col, (
            f"LAG_RETURN_COL_INDICES['{col}'] = {idx} but FEATURE_COLUMNS[{idx}] = {FEATURE_COLUMNS[idx]}"
        )


def test_target_column_is_adjclose():
    assert TARGET_COLUMN == 'adjClose'


def test_model_target_is_log_returns():
    assert MODEL_TARGET == 'log_returns'


def test_target_column_not_in_feature_columns():
    assert TARGET_COLUMN not in FEATURE_COLUMNS


def test_model_target_in_feature_columns():
    """log_returns is used both as a feature and as the prediction target."""
    assert MODEL_TARGET in FEATURE_COLUMNS


def test_test_size_fraction_valid():
    assert 0 < TEST_SIZE_FRACTION < 1


def test_min_train_rows_positive():
    assert MIN_TRAIN_ROWS > 0


def test_tscv_splits_at_least_two():
    assert TSCV_N_SPLITS >= 2


def test_forecast_days_positive():
    assert FORECAST_DAYS > 0


def test_ma_windows_are_positive_ints():
    assert all(isinstance(w, int) and w > 0 for w in MA_WINDOWS)


def test_base_dir_is_absolute():
    assert os.path.isabs(BASE_DIR)


def test_models_dir_is_absolute():
    assert os.path.isabs(MODELS_DIR)


def test_data_dir_is_absolute():
    assert os.path.isabs(DATA_DIR)
