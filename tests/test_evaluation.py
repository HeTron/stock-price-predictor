"""Tests for evaluate_model, cross_validate_model, compare_models, get_feature_importance."""
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from src.config import FEATURE_COLUMNS, TSCV_N_SPLITS
from src.trading_predictor import (
    evaluate_model,
    cross_validate_model,
    compare_models,
    get_feature_importance,
)


@pytest.fixture
def fitted_ridge(sample_stock_df):
    from sklearn.pipeline import make_pipeline
    X = sample_stock_df[FEATURE_COLUMNS].values
    y = sample_stock_df['adjClose'].values
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(X, y)
    return model, X, y


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------

class TestEvaluateModel:
    def test_returns_all_four_keys(self, fitted_ridge):
        model, X, y = fitted_ridge
        metrics = evaluate_model(model, X, y)
        for key in ('RMSE', 'MAE', 'R2', 'MAPE'):
            assert key in metrics, f"Missing key: {key}"

    def test_perfect_predictions(self):
        from sklearn.pipeline import make_pipeline
        # Trivial model: predict exactly y_train (overfit toy case)
        X = np.arange(20).reshape(-1, 1).astype(float)
        y = np.arange(20).astype(float) + 100  # keep y > 0 for MAPE

        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(X, y)
        # Use training data — perfect fit for linear data
        metrics = evaluate_model(model, X, y)
        assert metrics['RMSE'] < 1e-6
        assert metrics['MAE'] < 1e-6
        assert abs(metrics['R2'] - 1.0) < 1e-6
        assert metrics['MAPE'] < 1e-4

    def test_rmse_nonnegative(self, fitted_ridge):
        model, X, y = fitted_ridge
        metrics = evaluate_model(model, X, y)
        assert metrics['RMSE'] >= 0

    def test_mae_nonnegative(self, fitted_ridge):
        model, X, y = fitted_ridge
        metrics = evaluate_model(model, X, y)
        assert metrics['MAE'] >= 0

    def test_mae_leq_rmse(self, fitted_ridge):
        """MAE ≤ RMSE always holds (Jensen's inequality)."""
        model, X, y = fitted_ridge
        metrics = evaluate_model(model, X, y)
        assert metrics['MAE'] <= metrics['RMSE'] + 1e-9

    def test_returns_floats(self, fitted_ridge):
        model, X, y = fitted_ridge
        metrics = evaluate_model(model, X, y)
        for key, val in metrics.items():
            assert isinstance(val, float), f"{key} should be float, got {type(val)}"


# ---------------------------------------------------------------------------
# cross_validate_model
# ---------------------------------------------------------------------------

class TestCrossValidateModel:
    def test_returns_correct_n_folds(self, sample_stock_df):
        from sklearn.pipeline import make_pipeline
        X = sample_stock_df[FEATURE_COLUMNS].values
        y = sample_stock_df['adjClose'].values
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        n_splits = 3
        fold_metrics = cross_validate_model(model, X, y, n_splits=n_splits)
        for key in ('RMSE', 'MAE', 'R2', 'MAPE'):
            assert len(fold_metrics[key]) == n_splits, (
                f"Expected {n_splits} fold values for {key}, got {len(fold_metrics[key])}"
            )

    def test_all_rmse_nonnegative(self, sample_stock_df):
        from sklearn.pipeline import make_pipeline
        X = sample_stock_df[FEATURE_COLUMNS].values
        y = sample_stock_df['adjClose'].values
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        fold_metrics = cross_validate_model(model, X, y, n_splits=2)
        assert all(v >= 0 for v in fold_metrics['RMSE'])

    def test_returns_dict_with_all_keys(self, sample_stock_df):
        from sklearn.pipeline import make_pipeline
        X = sample_stock_df[FEATURE_COLUMNS].values
        y = sample_stock_df['adjClose'].values
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        fold_metrics = cross_validate_model(model, X, y, n_splits=2)
        for key in ('RMSE', 'MAE', 'R2', 'MAPE'):
            assert key in fold_metrics


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------

class TestCompareModels:
    def test_returns_pipeline_and_dataframe(self, sample_stock_df):
        X = sample_stock_df[FEATURE_COLUMNS].values
        y = sample_stock_df['adjClose'].values
        best_model, comparison_df = compare_models(X, y)
        assert isinstance(best_model, Pipeline)
        assert isinstance(comparison_df, pd.DataFrame)

    def test_comparison_df_has_three_models(self, sample_stock_df):
        X = sample_stock_df[FEATURE_COLUMNS].values
        y = sample_stock_df['adjClose'].values
        _, comparison_df = compare_models(X, y)
        assert len(comparison_df) == 3

    def test_comparison_df_sorted_by_rmse(self, sample_stock_df):
        X = sample_stock_df[FEATURE_COLUMNS].values
        y = sample_stock_df['adjClose'].values
        _, comparison_df = compare_models(X, y)
        rmse_values = comparison_df['RMSE_mean'].tolist()
        assert rmse_values == sorted(rmse_values), "comparison_df should be sorted by RMSE_mean ascending"

    def test_best_model_is_fitted(self, sample_stock_df):
        X = sample_stock_df[FEATURE_COLUMNS].values
        y = sample_stock_df['adjClose'].values
        best_model, _ = compare_models(X, y)
        # A fitted sklearn pipeline can predict
        preds = best_model.predict(X[:5])
        assert len(preds) == 5

    def test_model_names_in_comparison(self, sample_stock_df):
        X = sample_stock_df[FEATURE_COLUMNS].values
        y = sample_stock_df['adjClose'].values
        _, comparison_df = compare_models(X, y)
        model_names = set(comparison_df['Model'])
        assert model_names == {'Ridge', 'RandomForest', 'GradientBoosting'}


# ---------------------------------------------------------------------------
# get_feature_importance
# ---------------------------------------------------------------------------

class TestGetFeatureImportance:
    def test_returns_dataframe(self, fitted_ridge):
        model, _, _ = fitted_ridge
        fi_df = get_feature_importance(model)
        assert isinstance(fi_df, pd.DataFrame)

    def test_has_feature_and_importance_columns(self, fitted_ridge):
        model, _, _ = fitted_ridge
        fi_df = get_feature_importance(model)
        assert 'Feature' in fi_df.columns
        assert 'Importance' in fi_df.columns

    def test_all_features_present(self, fitted_ridge):
        model, _, _ = fitted_ridge
        fi_df = get_feature_importance(model)
        assert set(fi_df['Feature']) == set(FEATURE_COLUMNS)

    def test_sorted_descending(self, fitted_ridge):
        model, _, _ = fitted_ridge
        fi_df = get_feature_importance(model)
        importances = fi_df['Importance'].tolist()
        assert importances == sorted(importances, reverse=True), (
            "Feature importances should be sorted descending"
        )

    def test_importances_nonnegative(self, fitted_ridge):
        model, _, _ = fitted_ridge
        fi_df = get_feature_importance(model)
        assert (fi_df['Importance'] >= 0).all()

    def test_works_with_random_forest(self, sample_stock_df):
        from sklearn.pipeline import make_pipeline
        from sklearn.ensemble import RandomForestRegressor
        X = sample_stock_df[FEATURE_COLUMNS].values
        y = sample_stock_df['adjClose'].values
        model = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=10, random_state=42))
        model.fit(X, y)
        fi_df = get_feature_importance(model)
        assert set(fi_df['Feature']) == set(FEATURE_COLUMNS)
        assert (fi_df['Importance'] >= 0).all()
