import numpy as np
import pandas as pd

from ai.predict.run_prediction import (
    build_supervised_features,
    calculate_metrics,
    split_train_validation,
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
