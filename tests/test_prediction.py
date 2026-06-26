import sys
import types

import numpy as np
import pandas as pd

from ai.predict.run_prediction import (
    build_direct_horizon_features,
    build_supervised_features,
    arima_direct_forecast,
    calculate_metrics,
    naive_direct_forecast,
    select_common_example_origin,
    select_common_direct_evaluation_origins,
    select_direct_evaluation_windows,
    split_train_validation,
    validate_multi_horizon_config,
)


def test_split_train_validation_keeps_time_order():
    data = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=10),
            "close": range(10),
        }
    )

    train, validation = split_train_validation(data, validation_ratio=0.3)

    assert len(train) == 7
    assert len(validation) == 3
    assert train["date"].max() < validation["date"].min()


def test_build_supervised_features_uses_past_values_only():
    data = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=6),
            "open": [10, 11, 12, 13, 14, 15],
            "high": [11, 12, 13, 14, 15, 16],
            "low": [9, 10, 11, 12, 13, 14],
            "close": [10, 12, 14, 16, 18, 20],
            "volume": [100, 110, 120, 130, 140, 150],
        }
    )

    features = build_supervised_features(data, lags=(1, 2), rolling_windows=(2,))

    first = features.iloc[0]
    assert first["date"] == pd.Timestamp("2026-01-04")
    assert first["target_close"] == 16
    assert first["close_lag_1"] == 14
    assert first["close_lag_2"] == 12
    assert first["ma_2_lag_1"] == 13


def test_calculate_metrics_returns_expected_values():
    actual = np.array([100.0, 110.0, 90.0])
    predicted = np.array([100.0, 100.0, 100.0])

    metrics = calculate_metrics(actual, predicted)

    assert metrics["MAE"] == 20 / 3
    assert round(metrics["RMSE"], 6) == round(np.sqrt(200 / 3), 6)
    assert round(metrics["MAPE"], 6) == round(((0 / 100) + (10 / 110) + (10 / 90)) / 3, 6)


def test_direct_horizon_features_target_exact_future_close():
    data = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=8),
            "open": range(10, 18),
            "high": range(11, 19),
            "low": range(9, 17),
            "close": [10, 11, 12, 13, 14, 15, 16, 17],
            "volume": range(100, 108),
        }
    )

    features = build_direct_horizon_features(
        data, horizon=2, lags=(1,), rolling_windows=(2,)
    )

    first = features.iloc[0]
    assert first["origin_date"] == pd.Timestamp("2026-01-04")
    assert first["target_date"] == pd.Timestamp("2026-01-06")
    assert first["origin_close"] == 13
    assert first["target_close"] == 15
    assert round(first["target_return"], 6) == round(15 / 13 - 1, 6)
    assert first["close_lag_1"] == 12


def test_direct_horizon_features_never_use_rows_without_future_target():
    data = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=7),
            "open": range(10, 17),
            "high": range(11, 18),
            "low": range(9, 16),
            "close": range(10, 17),
            "volume": range(100, 107),
        }
    )

    features = build_direct_horizon_features(
        data, horizon=3, lags=(1,), rolling_windows=(2,)
    )

    assert features["target_date"].max() == pd.Timestamp("2026-01-07")
    assert (features["target_date"] > features["origin_date"]).all()


def test_direct_evaluation_windows_keep_train_targets_before_validation_origins():
    features = pd.DataFrame(
        {
            "origin_date": pd.date_range("2026-01-01", periods=10),
            "target_date": pd.date_range("2026-01-03", periods=10),
            "target_return": np.linspace(0.01, 0.10, 10),
        }
    )

    train, validation = select_direct_evaluation_windows(features, evaluation_points=3)

    assert len(validation) == 3
    assert train["target_date"].max() < validation["origin_date"].min()
    assert train["origin_date"].max() < validation["origin_date"].min()


def test_naive_direct_forecast_uses_origin_close_for_each_horizon():
    validation = pd.DataFrame({"origin_close": [100.0, 105.5, 99.0]})

    predicted = naive_direct_forecast(validation)

    np.testing.assert_allclose(predicted, [100.0, 105.5, 99.0])


def test_select_common_example_origin_has_all_horizons():
    rows = []
    for horizon in (1, 5, 20, 60):
        for model in ("Naive", "ARIMA", "XGBoost", "LSTM"):
            rows.append(
                {
                    "model": model,
                    "horizon": horizon,
                    "origin_date": pd.Timestamp("2026-01-10"),
                    "target_date": pd.Timestamp("2026-01-10") + pd.Timedelta(days=horizon),
                }
            )
    predictions = pd.DataFrame(rows)

    assert select_common_example_origin(predictions, (1, 5, 20, 60)) == pd.Timestamp("2026-01-10")


def test_multi_horizon_config_requires_positive_evaluation_points():
    with np.testing.assert_raises_regex(ValueError, "正整数"):
        validate_multi_horizon_config((1, 5, 20, 60), evaluation_points=0)


def test_common_direct_evaluation_origins_align_all_horizons():
    frames = {
        1: pd.DataFrame({"origin_date": pd.date_range("2026-01-01", periods=10)}),
        5: pd.DataFrame({"origin_date": pd.date_range("2026-01-01", periods=8)}),
        20: pd.DataFrame({"origin_date": pd.date_range("2026-01-01", periods=9)}),
    }

    origins = select_common_direct_evaluation_origins(frames, evaluation_points=3)

    assert list(origins) == list(pd.date_range("2026-01-06", periods=3))


def test_arima_direct_forecast_fits_once_then_appends_observed_history(monkeypatch):
    class FakeResult:
        append_calls = 0

        def __init__(self, history):
            self.history = np.asarray(history, dtype=float)

        def append(self, values, refit):
            assert refit is False
            FakeResult.append_calls += 1
            return FakeResult(np.concatenate([self.history, np.asarray(values, dtype=float)]))

        def forecast(self, steps):
            return np.repeat(self.history[-1], steps)

    class FakeARIMA:
        fit_calls = 0

        def __init__(self, history, order):
            self.history = history

        def fit(self):
            FakeARIMA.fit_calls += 1
            return FakeResult(self.history)

    statsmodels = types.ModuleType("statsmodels")
    tsa = types.ModuleType("statsmodels.tsa")
    arima = types.ModuleType("statsmodels.tsa.arima")
    model = types.ModuleType("statsmodels.tsa.arima.model")
    model.ARIMA = FakeARIMA
    monkeypatch.setitem(sys.modules, "statsmodels", statsmodels)
    monkeypatch.setitem(sys.modules, "statsmodels.tsa", tsa)
    monkeypatch.setitem(sys.modules, "statsmodels.tsa.arima", arima)
    monkeypatch.setitem(sys.modules, "statsmodels.tsa.arima.model", model)

    prices = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=6),
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        }
    )
    validation = pd.DataFrame(
        {
            "origin_date": [pd.Timestamp("2026-01-03"), pd.Timestamp("2026-01-04")],
        }
    )

    predicted = arima_direct_forecast(prices, validation, horizon=2)

    assert len(predicted) == 2
    assert FakeARIMA.fit_calls == 1
    assert FakeResult.append_calls == 1
