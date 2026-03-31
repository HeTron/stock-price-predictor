"""Tests for src/trading_predictor.py — no API calls; uses synthetic fixtures."""
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import FEATURE_COLUMNS, TARGET_COLUMN, MODEL_TARGET, TEST_SIZE_FRACTION
from src.trading_predictor import (
    calculate_rsi,
    split_time_series,
    training_data_prep,
    walk_forward_backtest,
)


# ---------------------------------------------------------------------------
# calculate_rsi
# ---------------------------------------------------------------------------

class TestCalculateRsi:
    def test_output_length_matches_input(self, sample_stock_df):
        rsi = calculate_rsi(sample_stock_df)
        assert len(rsi) == len(sample_stock_df)

    def test_non_nan_values_in_range(self, sample_stock_df):
        rsi = calculate_rsi(sample_stock_df).fillna(50)
        assert rsi.between(0, 100).all(), "RSI values must be in [0, 100]"

    def test_fillna_50_leaves_no_nan(self, sample_stock_df):
        rsi = calculate_rsi(sample_stock_df).fillna(50)
        assert not rsi.isna().any()


# ---------------------------------------------------------------------------
# split_time_series
# ---------------------------------------------------------------------------

class TestSplitTimeSeries:
    def test_preserves_temporal_order(self, sample_stock_df):
        train_df, test_df = split_time_series(sample_stock_df)
        assert train_df['date'].max() <= test_df['date'].min()

    def test_no_overlap(self, sample_stock_df):
        train_df, test_df = split_time_series(sample_stock_df)
        assert set(train_df['date']).isdisjoint(set(test_df['date']))

    def test_proportions_approximately_correct(self, sample_stock_df):
        n = len(sample_stock_df)
        _, test_df = split_time_series(sample_stock_df)
        actual_fraction = len(test_df) / n
        assert abs(actual_fraction - TEST_SIZE_FRACTION) <= 1 / n + 0.01

    def test_full_coverage(self, sample_stock_df):
        train_df, test_df = split_time_series(sample_stock_df)
        assert len(train_df) + len(test_df) == len(sample_stock_df)

    def test_custom_test_size(self, sample_stock_df):
        _, test_df = split_time_series(sample_stock_df, test_size=0.3)
        expected = int(len(sample_stock_df) * 0.3)
        assert abs(len(test_df) - expected) <= 1


# ---------------------------------------------------------------------------
# training_data_prep
# ---------------------------------------------------------------------------

class TestTrainingDataPrep:
    def test_x_has_correct_number_of_columns(self, sample_stock_df):
        X, _ = training_data_prep(sample_stock_df)
        assert X.shape[1] == len(FEATURE_COLUMNS)

    def test_y_length_matches_x(self, sample_stock_df):
        X, y = training_data_prep(sample_stock_df)
        assert len(X) == len(y)

    def test_no_nan_in_x(self, sample_stock_df):
        X, _ = training_data_prep(sample_stock_df)
        assert not np.isnan(X).any(), "X contains NaN values"

    def test_no_nan_in_y(self, sample_stock_df):
        _, y = training_data_prep(sample_stock_df)
        assert not np.isnan(y).any(), "y contains NaN values"

    def test_x_is_numpy_array(self, sample_stock_df):
        X, _ = training_data_prep(sample_stock_df)
        assert isinstance(X, np.ndarray)

    def test_y_is_numpy_array(self, sample_stock_df):
        _, y = training_data_prep(sample_stock_df)
        assert isinstance(y, np.ndarray)

    def test_y_is_log_returns_not_adjclose(self, sample_stock_df):
        """y must be log returns so the model is scale-invariant across price regimes."""
        _, y = training_data_prep(sample_stock_df)
        expected = sample_stock_df[MODEL_TARGET].values
        np.testing.assert_array_equal(y, expected)

    def test_y_is_not_adjclose(self, sample_stock_df):
        """Regression guard: y must NOT be adjClose (that caused price-regime anchoring)."""
        _, y = training_data_prep(sample_stock_df)
        adjclose = sample_stock_df[TARGET_COLUMN].values
        assert not np.allclose(y, adjclose), (
            "y should be log_returns, not adjClose — "
            "predicting raw prices anchors the model to historical price regimes"
        )

    def test_feature_column_order(self, sample_stock_df):
        X, _ = training_data_prep(sample_stock_df)
        expected = sample_stock_df[FEATURE_COLUMNS].values
        np.testing.assert_array_equal(X, expected)

    def test_lag_return_features_present(self, sample_stock_df):
        """Return-based lag features must be used, not absolute price lags."""
        for col in ['lag_return_1', 'lag_return_5', 'lag_return_30', 'lag_return_45']:
            assert col in FEATURE_COLUMNS, f"{col} missing from FEATURE_COLUMNS"


# ---------------------------------------------------------------------------
# walk_forward_backtest
# ---------------------------------------------------------------------------

class TestWalkForwardBacktest:
    def test_returns_dataframe(self, sample_stock_df):
        result = walk_forward_backtest(sample_stock_df, n_splits=2)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, sample_stock_df):
        result = walk_forward_backtest(sample_stock_df, n_splits=2)
        for col in ('Date', 'Actual', 'Predicted', 'AbsError'):
            assert col in result.columns, f"Missing column: {col}"

    def test_results_are_in_price_space(self, sample_stock_df):
        """Actual and Predicted should be in price space (~100), not return space (~0)."""
        result = walk_forward_backtest(sample_stock_df, n_splits=2)
        assert result['Actual'].mean() > 1.0, "Actual values look like returns, not prices"
        assert result['Predicted'].mean() > 1.0, "Predicted values look like returns, not prices"

    def test_abs_error_matches_difference(self, sample_stock_df):
        result = walk_forward_backtest(sample_stock_df, n_splits=2)
        computed = (result['Actual'] - result['Predicted']).abs()
        pd.testing.assert_series_equal(computed, result['AbsError'], check_names=False, rtol=1e-5)

    def test_no_nan_in_results(self, sample_stock_df):
        result = walk_forward_backtest(sample_stock_df, n_splits=2)
        assert not result[['Actual', 'Predicted', 'AbsError']].isna().any().any()

    def test_result_is_sorted_by_date(self, sample_stock_df):
        result = walk_forward_backtest(sample_stock_df, n_splits=2)
        dates = list(result['Date'])
        assert dates == sorted(dates)

    def test_no_future_data_leakage(self, sample_stock_df):
        from sklearn.model_selection import TimeSeriesSplit
        X = sample_stock_df[FEATURE_COLUMNS].values
        dates = np.array(sample_stock_df['date'].values)
        for train_idx, test_idx in TimeSeriesSplit(n_splits=2).split(X):
            assert set(dates[train_idx]).isdisjoint(set(dates[test_idx]))
