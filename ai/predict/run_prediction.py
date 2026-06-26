"""离线金价预测实验流水线。

运行后会在 analysis/output/prediction/ 下生成：
- prediction_metrics.csv：各模型 MAE/RMSE/MAPE 对比
- prediction_results.csv：验证集真实值与各模型预测值
- prediction_comparison.png：真实值 vs 模型预测曲线
- prediction_error.png：各模型绝对误差曲线

该模块用于 PPT 汇报，不接入 Streamlit 页面，避免把实验预测误解为实时投资建议。
"""

from __future__ import annotations

import argparse
import math
import random
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.clean_data import clean_price_data
from analysis.load_data import load_price_data
from crawler.config import DEFAULT_DB_PATH, PROJECT_ROOT


OUTPUT_DIR = PROJECT_ROOT / "analysis" / "output" / "prediction"
DEFAULT_LAGS = (1, 2, 3, 5, 10, 20)
DEFAULT_ROLLING_WINDOWS = (5, 10, 20, 60)
DEFAULT_MULTI_HORIZONS = (1, 3, 5, 20, 60)
DEFAULT_SHORT_HORIZONS = (1, 3, 5)


def set_random_seed(seed: int = 42) -> None:
    """固定随机种子，尽量让每次实验结果可复现。"""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def load_prediction_data(db_path: str | Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """读取并清洗金价数据。"""

    prices = clean_price_data(load_price_data(db_path))
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in prices.columns]
    if missing:
        raise ValueError(f"缺少预测所需字段: {missing}")
    return prices[required].dropna(subset=["date", "close"]).reset_index(drop=True)


def split_train_validation(
    data: pd.DataFrame,
    validation_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按时间顺序切分训练集和验证集，避免未来数据泄漏。"""

    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio 必须在 0 和 1 之间")

    split_index = int(len(data) * (1 - validation_ratio))
    if split_index <= 0 or split_index >= len(data):
        raise ValueError("数据量不足，无法按当前比例切分训练集和验证集")

    train = data.iloc[:split_index].copy().reset_index(drop=True)
    validation = data.iloc[split_index:].copy().reset_index(drop=True)
    return train, validation


def build_supervised_features(
    data: pd.DataFrame,
    lags: Iterable[int] = DEFAULT_LAGS,
    rolling_windows: Iterable[int] = DEFAULT_ROLLING_WINDOWS,
) -> pd.DataFrame:
    """构造机器学习监督学习特征。

    所有特征都使用 lag 或 shift(1)，确保预测当天收盘价时不会看到当天结果。
    """

    working = data.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    numeric_columns = ["open", "high", "low", "close", "volume"]
    for column in numeric_columns:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    returns = working["close"].pct_change()
    features = pd.DataFrame(
        {
            "date": working["date"],
            "target_close": working["close"],
            "target_return": returns,
            "previous_close": working["close"].shift(1),
        }
    )
    for lag in lags:
        features[f"close_lag_{lag}"] = working["close"].shift(lag)
        features[f"return_lag_{lag}"] = working["close"].pct_change().shift(lag)
        if "volume" in working.columns:
            features[f"volume_lag_{lag}"] = working["volume"].shift(lag)

    for window in rolling_windows:
        features[f"ma_{window}_lag_1"] = working["close"].rolling(window).mean().shift(1)
        features[f"volatility_{window}_lag_1"] = (
            working["close"].pct_change().rolling(window).std().shift(1)
        )
        features[f"range_{window}_lag_1"] = (
            (working["high"] - working["low"]).rolling(window).mean().shift(1)
            if {"high", "low"}.issubset(working.columns)
            else np.nan
        )

    return features.dropna().reset_index(drop=True)


def build_direct_horizon_features(
    data: pd.DataFrame,
    horizon: int,
    lags: Iterable[int] = DEFAULT_LAGS,
    rolling_windows: Iterable[int] = DEFAULT_ROLLING_WINDOWS,
) -> pd.DataFrame:
    """构造直接预测第 ``horizon`` 个交易日后价格的特征。

    每一行代表一个预测起点。特征只使用起点及其之前的数据，
    ``target_close`` 是起点之后第 ``horizon`` 个交易日的真实收盘价。
    """

    if horizon <= 0:
        raise ValueError("horizon 必须是正整数")

    working = data.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    numeric_columns = ["open", "high", "low", "close", "volume"]
    for column in numeric_columns:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    features = pd.DataFrame(
        {
            "origin_date": working["date"],
            "target_date": working["date"].shift(-horizon),
            "origin_close": working["close"],
            "target_close": working["close"].shift(-horizon),
        }
    )
    features["target_return"] = features["target_close"] / features["origin_close"] - 1

    for lag in lags:
        features[f"close_lag_{lag}"] = working["close"].shift(lag)
        features[f"return_lag_{lag}"] = working["close"].pct_change().shift(lag)
        if "volume" in working.columns:
            features[f"volume_lag_{lag}"] = working["volume"].shift(lag)

    for window in rolling_windows:
        features[f"ma_{window}_lag_1"] = working["close"].rolling(window).mean().shift(1)
        features[f"volatility_{window}_lag_1"] = (
            working["close"].pct_change().rolling(window).std().shift(1)
        )
        features[f"range_{window}_lag_1"] = (
            (working["high"] - working["low"]).rolling(window).mean().shift(1)
            if {"high", "low"}.issubset(working.columns)
            else np.nan
        )

    return features.dropna().reset_index(drop=True)


def select_direct_evaluation_windows(
    features: pd.DataFrame,
    evaluation_points: int,
    validation_origins: pd.DatetimeIndex | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按预测起点划分直接预测训练集与验证集，避免标签跨界。"""

    if evaluation_points <= 0:
        raise ValueError("evaluation_points 必须是正整数")
    if len(features) <= evaluation_points:
        raise ValueError("特征样本不足，无法划分训练集和验证集")

    ordered = features.sort_values("origin_date").reset_index(drop=True)
    if validation_origins is None:
        validation = ordered.tail(evaluation_points).reset_index(drop=True)
    else:
        validation = ordered[
            ordered["origin_date"].isin(pd.to_datetime(validation_origins))
        ].reset_index(drop=True)
        if len(validation) != evaluation_points:
            raise ValueError("指定的共同验证预测起点未被当前步长完整覆盖")
    validation_start = validation["origin_date"].iloc[0]
    train = ordered[
        (ordered["origin_date"] < validation_start)
        & (ordered["target_date"] < validation_start)
    ].reset_index(drop=True)
    if train.empty:
        raise ValueError("训练样本不足：预测标签与验证起点没有可用间隔")
    return train, validation


def select_common_direct_evaluation_origins(
    features_by_horizon: dict[int, pd.DataFrame],
    evaluation_points: int,
) -> pd.DatetimeIndex:
    """选择所有预测步长共同拥有的最后一组历史预测起点。"""

    if evaluation_points <= 0:
        raise ValueError("evaluation_points 必须是正整数")
    origin_sets = [set(pd.to_datetime(features["origin_date"])) for features in features_by_horizon.values()]
    if not origin_sets:
        raise ValueError("没有可用于选择共同预测起点的特征数据")
    common_origins = sorted(set.intersection(*origin_sets))
    if len(common_origins) < evaluation_points:
        raise ValueError("共同预测起点不足，无法完成多预测步长回测")
    return pd.DatetimeIndex(common_origins[-evaluation_points:])


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """计算预测误差指标。"""

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    non_zero = actual != 0
    mape = float(np.mean(np.abs(error[non_zero] / actual[non_zero]))) if non_zero.any() else math.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def naive_forecast(train: pd.DataFrame, validation: pd.DataFrame) -> np.ndarray:
    """朴素基线：下一日价格等于上一交易日收盘价。"""

    combined_close = pd.concat([train["close"], validation["close"]], ignore_index=True)
    predictions = combined_close.shift(1).iloc[len(train) :].to_numpy(dtype=float)
    predictions[0] = float(train["close"].iloc[-1])
    return predictions


def naive_direct_forecast(validation: pd.DataFrame) -> np.ndarray:
    """直接预测基线：未来任意步长价格等于预测起点收盘价。"""

    return validation["origin_close"].to_numpy(dtype=float)


def _direct_feature_columns(data: pd.DataFrame) -> list[str]:
    metadata_columns = {
        "origin_date",
        "target_date",
        "origin_close",
        "target_close",
        "target_return",
    }
    return [column for column in data.columns if column not in metadata_columns]


def arima_direct_forecast(
    prices: pd.DataFrame,
    validation: pd.DataFrame,
    horizon: int,
    order: tuple[int, int, int] = (5, 1, 0),
) -> np.ndarray:
    """在每个预测起点直接预测第 ``horizon`` 个交易日后的价格。"""

    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:
        raise RuntimeError("缺少 statsmodels，请先安装 statsmodels") from exc

    ordered_prices = prices.sort_values("date").reset_index(drop=True)
    origins = pd.to_datetime(validation["origin_date"]).sort_values().to_list()
    first_origin = origins[0]
    history = ordered_prices.loc[
        ordered_prices["date"] <= first_origin, "close"
    ].to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = ARIMA(history, order=order).fit()

    predictions: list[float] = []
    previous_origin = first_origin
    for index, origin_date in enumerate(origins):
        if index:
            observed = ordered_prices.loc[
                (ordered_prices["date"] > previous_origin)
                & (ordered_prices["date"] <= origin_date),
                "close",
            ].to_numpy(dtype=float)
            if observed.size:
                fitted = fitted.append(observed, refit=False)
        predictions.append(float(np.asarray(fitted.forecast(steps=horizon))[-1]))
        previous_origin = origin_date
    return np.array(predictions, dtype=float)


def xgboost_direct_forecast(train: pd.DataFrame, validation: pd.DataFrame) -> np.ndarray:
    """按单个预测步长训练 XGBoost，输出未来目标日收盘价。"""

    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError("缺少 xgboost，请先安装 xgboost") from exc

    feature_columns = _direct_feature_columns(train)
    model = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
    )
    model.fit(train[feature_columns], train["target_return"])
    predicted_returns = model.predict(validation[feature_columns])
    return validation["origin_close"].to_numpy(dtype=float) * (1 + predicted_returns)


def _build_direct_lstm_samples(
    prices: pd.DataFrame,
    horizon: int,
    lookback: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """构造以预测起点结束的收益率窗口及其未来目标。"""

    ordered = prices.sort_values("date").reset_index(drop=True)
    closes = ordered["close"].to_numpy(dtype=float)
    returns = pd.Series(closes).pct_change().fillna(0.0).to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    windows: list[np.ndarray] = []
    for origin_index in range(lookback, len(ordered) - horizon):
        target_index = origin_index + horizon
        origin_close = closes[origin_index]
        target_close = closes[target_index]
        rows.append(
            {
                "origin_date": ordered["date"].iloc[origin_index],
                "target_date": ordered["date"].iloc[target_index],
                "origin_close": origin_close,
                "target_close": target_close,
                "target_return": target_close / origin_close - 1,
            }
        )
        windows.append(returns[origin_index - lookback + 1 : origin_index + 1])
    return pd.DataFrame(rows), np.array(windows, dtype=np.float32)


def lstm_direct_forecast(
    prices: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    horizon: int,
    lookback: int = 60,
    epochs: int = 60,
    batch_size: int = 32,
) -> np.ndarray:
    """按单个预测步长训练 LSTM，输出未来目标日收盘价。"""

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("缺少 torch，请先安装 PyTorch") from exc

    set_random_seed(42)
    sample_meta, sample_windows = _build_direct_lstm_samples(prices, horizon, lookback)
    train_dates = set(pd.to_datetime(train["origin_date"]))
    validation_dates = set(pd.to_datetime(validation["origin_date"]))
    train_mask = sample_meta["origin_date"].isin(train_dates).to_numpy()
    validation_mask = sample_meta["origin_date"].isin(validation_dates).to_numpy()

    if not train_mask.any() or not validation_mask.any():
        raise ValueError("LSTM 样本不足，无法对齐直接预测训练集和验证集")

    x_train_raw = sample_windows[train_mask]
    x_validation_raw = sample_windows[validation_mask]
    y_train_raw = sample_meta.loc[train_mask, "target_return"].to_numpy(dtype=np.float32)
    mean = float(x_train_raw.mean())
    std = float(x_train_raw.std()) or 1.0
    target_mean = float(y_train_raw.mean())
    target_std = float(y_train_raw.std()) or 1.0

    x_train = torch.tensor((x_train_raw - mean) / std).unsqueeze(-1)
    y_train = torch.tensor((y_train_raw - target_mean) / target_std).unsqueeze(-1)
    x_validation = torch.tensor((x_validation_raw - mean) / std).unsqueeze(-1)

    class GoldDirectLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
            self.head = nn.Linear(64, 1)

        def forward(self, x):
            output, _ = self.lstm(x)
            return self.head(output[:, -1, :])

    model = GoldDirectLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=False)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        predicted_returns = model(x_validation).squeeze(-1).numpy() * target_std + target_mean

    validation_meta = sample_meta.loc[validation_mask].sort_values("origin_date")
    predicted_by_date = pd.Series(
        predicted_returns,
        index=pd.to_datetime(validation_meta["origin_date"]),
    )
    ordered_returns = predicted_by_date.reindex(pd.to_datetime(validation["origin_date"])).to_numpy(dtype=float)
    return validation["origin_close"].to_numpy(dtype=float) * (1 + ordered_returns)


def run_multi_horizon_experiment(
    db_path: str | Path = DEFAULT_DB_PATH,
    horizons: tuple[int, ...] = DEFAULT_MULTI_HORIZONS,
    evaluation_points: int = 120,
    lstm_epochs: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """运行 Naive、ARIMA、XGBoost、LSTM 的直接多预测步长历史回测。"""

    set_random_seed(42)
    validate_multi_horizon_config(horizons, evaluation_points)
    prices = load_prediction_data(db_path)
    features_by_horizon = {
        horizon: build_direct_horizon_features(prices, horizon=horizon)
        for horizon in horizons
    }
    common_validation_origins = select_common_direct_evaluation_origins(
        features_by_horizon,
        evaluation_points=evaluation_points,
    )
    prediction_frames: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, object]] = []

    for horizon in horizons:
        train, validation = select_direct_evaluation_windows(
            features_by_horizon[horizon],
            evaluation_points,
            validation_origins=common_validation_origins,
        )
        predictions = {
            "Naive": naive_direct_forecast(validation),
            "ARIMA": arima_direct_forecast(prices, validation, horizon=horizon),
            "XGBoost": xgboost_direct_forecast(train, validation),
            "LSTM": lstm_direct_forecast(
                prices,
                train,
                validation,
                horizon=horizon,
                epochs=lstm_epochs,
            ),
        }

        for model_name, predicted_close in predictions.items():
            actual_close = validation["target_close"].to_numpy(dtype=float)
            metrics_rows.append(
                {
                    "model": model_name,
                    "horizon": horizon,
                    **calculate_metrics(actual_close, predicted_close),
                    "train_start": train["origin_date"].iloc[0].strftime("%Y-%m-%d"),
                    "train_end": train["target_date"].iloc[-1].strftime("%Y-%m-%d"),
                    "validation_origin_start": validation["origin_date"].iloc[0].strftime("%Y-%m-%d"),
                    "validation_origin_end": validation["origin_date"].iloc[-1].strftime("%Y-%m-%d"),
                    "validation_samples": len(validation),
                }
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "model": model_name,
                        "horizon": horizon,
                        "origin_date": validation["origin_date"].to_numpy(),
                        "target_date": validation["target_date"].to_numpy(),
                        "origin_close": validation["origin_close"].to_numpy(dtype=float),
                        "actual_close": actual_close,
                        "predicted_close": predicted_close,
                    }
                )
            )

    return pd.concat(prediction_frames, ignore_index=True), pd.DataFrame(metrics_rows)


def arima_forecast(train: pd.DataFrame, validation: pd.DataFrame, order=(5, 1, 0)) -> np.ndarray:
    """ARIMA 滚动预测。"""

    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:
        raise RuntimeError("缺少 statsmodels，请先安装 statsmodels") from exc

    history = list(pd.to_numeric(train["close"], errors="coerce").dropna())
    predictions: list[float] = []
    for actual in validation["close"]:
        model = ARIMA(history, order=order)
        fitted = model.fit()
        prediction = float(fitted.forecast(steps=1)[0])
        predictions.append(prediction)
        history.append(float(actual))
    return np.array(predictions, dtype=float)


def xgboost_forecast(feature_data: pd.DataFrame, train_end_date: pd.Timestamp) -> np.ndarray:
    """使用 XGBoost 基于滞后特征预测验证集收盘价。"""

    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError("缺少 xgboost，请先安装 xgboost") from exc

    feature_columns = [
        column
        for column in feature_data.columns
        if column not in {"date", "target_close", "target_return", "previous_close"}
    ]
    train_mask = feature_data["date"] <= train_end_date
    validation_mask = feature_data["date"] > train_end_date

    model = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
    )
    model.fit(feature_data.loc[train_mask, feature_columns], feature_data.loc[train_mask, "target_return"])
    predicted_returns = model.predict(feature_data.loc[validation_mask, feature_columns])
    previous_close = feature_data.loc[validation_mask, "previous_close"].to_numpy(dtype=float)
    return previous_close * (1 + predicted_returns)


def _scale_series(train_values: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """用训练集均值方差标准化，避免验证集泄漏。"""

    mean = float(np.mean(train_values))
    std = float(np.std(train_values)) or 1.0
    return (values - mean) / std, mean, std


def _build_lstm_windows(values: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """构建 LSTM 滑动窗口。"""

    xs: list[np.ndarray] = []
    ys: list[float] = []
    for index in range(lookback, len(values)):
        xs.append(values[index - lookback : index])
        ys.append(values[index])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def lstm_forecast(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    lookback: int = 60,
    epochs: int = 60,
    batch_size: int = 32,
) -> np.ndarray:
    """训练轻量 LSTM，并预测验证集收盘价。"""

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("缺少 torch，请先安装 PyTorch") from exc

    set_random_seed(42)
    train_close = train["close"].to_numpy(dtype=float)
    all_close = pd.concat([train["close"], validation["close"]], ignore_index=True).to_numpy(dtype=float)
    all_returns = pd.Series(all_close).pct_change().fillna(0.0).to_numpy(dtype=float)
    train_returns = pd.Series(train_close).pct_change().fillna(0.0).to_numpy(dtype=float)
    scaled_all, mean, std = _scale_series(train_returns, all_returns)

    x_all, y_all = _build_lstm_windows(scaled_all, lookback)
    dates = pd.concat([train["date"], validation["date"]], ignore_index=True).iloc[lookback:].reset_index(drop=True)
    train_end_date = pd.to_datetime(train["date"].iloc[-1])
    train_mask = dates <= train_end_date
    validation_mask = dates > train_end_date

    x_train = torch.tensor(x_all[train_mask.to_numpy()]).unsqueeze(-1)
    y_train = torch.tensor(y_all[train_mask.to_numpy()]).unsqueeze(-1)
    x_validation = torch.tensor(x_all[validation_mask.to_numpy()]).unsqueeze(-1)

    class GoldLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
            self.head = nn.Linear(64, 1)

        def forward(self, x):
            output, _ = self.lstm(x)
            return self.head(output[:, -1, :])

    model = GoldLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=False)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        scaled_predictions = model(x_validation).squeeze(-1).numpy()
    predicted_returns = scaled_predictions * std + mean
    validation_previous_close = (
        pd.concat([train["close"], validation["close"]], ignore_index=True)
        .shift(1)
        .iloc[len(train) :]
        .to_numpy(dtype=float)
    )
    validation_previous_close[0] = float(train["close"].iloc[-1])
    return validation_previous_close * (1 + predicted_returns)


def plot_prediction_comparison(results: pd.DataFrame, output_path: Path) -> None:
    """绘制真实值与预测值对比图。"""

    plt.figure(figsize=(13, 6))
    plt.plot(results["date"], results["actual_close"], label="Actual", linewidth=2.2, color="#111827")
    color_map = [
        ("Naive", "#64748b"),
        ("ARIMA", "#2563eb"),
        ("XGBoost", "#059669"),
        ("LSTM", "#dc2626"),
    ]
    for column, color in color_map:
        if column in results.columns:
            plt.plot(results["date"], results[column], label=column, linewidth=1.6, alpha=0.9, color=color)
    plt.title("Gold Price Forecast: Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_prediction_error(results: pd.DataFrame, output_path: Path) -> None:
    """绘制各模型绝对误差曲线。"""

    plt.figure(figsize=(13, 5))
    color_map = [
        ("Naive", "#64748b"),
        ("ARIMA", "#2563eb"),
        ("XGBoost", "#059669"),
        ("LSTM", "#dc2626"),
    ]
    for column, color in color_map:
        if column in results.columns:
            plt.plot(
                results["date"],
                np.abs(results[column] - results["actual_close"]),
                label=column,
                linewidth=1.4,
                alpha=0.9,
                color=color,
            )
    plt.title("Forecast Absolute Error")
    plt.xlabel("Date")
    plt.ylabel("Absolute Error")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_metrics_bar(metrics: pd.DataFrame, output_path: Path) -> None:
    """绘制 MAE/RMSE/MAPE 指标对比柱状图。"""

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    metrics = metrics.copy()
    metrics["MAPE_percent"] = metrics["MAPE"] * 100
    plots = [("MAE", "MAE"), ("RMSE", "RMSE"), ("MAPE_percent", "MAPE (%)")]
    colors = ["#64748b", "#2563eb", "#059669", "#dc2626"]
    for ax, (column, title) in zip(axes, plots):
        ax.bar(metrics["model"], metrics[column], color=colors[: len(metrics)])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=25)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.set_ylim(top=max(metrics[column]) * 1.18)
        for index, value in enumerate(metrics[column]):
            ax.text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    figure.suptitle("Forecast Metrics Comparison")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_multi_horizon_metrics(metrics: pd.DataFrame, output_path: Path) -> None:
    """绘制多预测步长下的 MAE 与 MAPE 分组柱状图。"""

    horizons = sorted(metrics["horizon"].unique())
    models = [model for model in ["Naive", "ARIMA", "XGBoost", "LSTM"] if model in set(metrics["model"])]
    colors = {"Naive": "#64748b", "ARIMA": "#2563eb", "XGBoost": "#059669", "LSTM": "#dc2626"}
    x = np.arange(len(horizons))
    width = 0.78 / len(models)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    for axis, column, label in zip(axes, ["MAE", "MAPE"], ["MAE", "MAPE (%)"]):
        for index, model in enumerate(models):
            values = (
                metrics[metrics["model"] == model]
                .set_index("horizon")
                .reindex(horizons)[column]
                .to_numpy(dtype=float)
            )
            if column == "MAPE":
                values = values * 100
            positions = x - 0.39 + width / 2 + index * width
            axis.bar(positions, values, width=width, label=model, color=colors[model])
        axis.set_title(label)
        axis.set_xlabel("Forecast Horizon (trading days)")
        axis.set_xticks(x, [str(horizon) for horizon in horizons])
        axis.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Price Error")
    axes[1].set_ylabel("Percentage Error")
    axes[1].legend(loc="upper left")
    figure.suptitle("Direct Multi-Horizon Forecast Error")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def select_common_example_origin(
    predictions: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_MULTI_HORIZONS,
) -> pd.Timestamp:
    """选择同时拥有全部模型和预测步长的最近预测起点。"""

    expected_models = set(predictions["model"])
    required_horizons = set(horizons)
    eligible: list[pd.Timestamp] = []
    for origin_date, group in predictions.groupby("origin_date"):
        if set(group["horizon"]) != required_horizons:
            continue
        if all(
            set(group.loc[group["horizon"] == horizon, "model"]) == expected_models
            for horizon in required_horizons
        ):
            eligible.append(pd.Timestamp(origin_date))
    if not eligible:
        raise ValueError("没有同时覆盖全部预测步长和模型的示例预测起点")
    return max(eligible)


def select_short_horizon_predictions(
    predictions: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_SHORT_HORIZONS,
) -> pd.DataFrame:
    """筛选用于短期走势展示的直接预测记录。"""

    return predictions[predictions["horizon"].isin(horizons)].copy()


def select_short_horizon_metrics(
    metrics: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_SHORT_HORIZONS,
) -> pd.DataFrame:
    """筛选用于短期 MAPE 对比图的汇总指标。"""

    return metrics[metrics["horizon"].isin(horizons)].copy()


def validate_multi_horizon_config(
    horizons: tuple[int, ...],
    evaluation_points: int,
) -> None:
    """确保各预测步长的验证窗口存在共同的示例预测起点。"""

    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("horizons 必须包含正整数预测步长")
    if evaluation_points <= 0:
        raise ValueError("evaluation_points 必须是正整数")


def plot_multi_horizon_example(
    prices: pd.DataFrame,
    predictions: pd.DataFrame,
    example_origin: pd.Timestamp,
    output_path: Path,
) -> None:
    """绘制固定起点的未来 60 个交易日真实路径及各模型预测终点。"""

    example_rows = predictions[predictions["origin_date"] == example_origin].copy()
    max_horizon = int(example_rows["horizon"].max())
    ordered_prices = prices.sort_values("date").reset_index(drop=True)
    origin_index = ordered_prices.index[ordered_prices["date"] == example_origin]
    if origin_index.empty:
        raise ValueError("示例预测起点不在价格数据中")
    path = ordered_prices.iloc[origin_index[0] : origin_index[0] + max_horizon + 1]

    colors = {"Naive": "#64748b", "ARIMA": "#2563eb", "XGBoost": "#059669", "LSTM": "#dc2626"}
    figure, axis = plt.subplots(figsize=(13, 5.6))
    axis.plot(path["date"], path["close"], color="#111827", linewidth=2.2, label="Actual close")
    axis.scatter(
        [example_origin],
        [path["close"].iloc[0]],
        color="#111827",
        marker="o",
        s=55,
        zorder=5,
        label="Forecast origin",
    )
    for model, group in example_rows.groupby("model"):
        axis.scatter(
            group["target_date"],
            group["predicted_close"],
            color=colors.get(model, "#0f172a"),
            marker="X",
            s=70,
            zorder=6,
            label=f"{model} direct forecast",
        )
    axis.set_title("Direct Forecast Endpoints From One Origin")
    axis.set_xlabel("Target Date")
    axis.set_ylabel("Close Price")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=9)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_short_horizon_trajectories(
    prices: pd.DataFrame,
    predictions: pd.DataFrame,
    output_path: Path,
    horizons: tuple[int, ...] = DEFAULT_SHORT_HORIZONS,
    display_points: int = 60,
) -> None:
    """按模型分面绘制 t+1、t+3、t+5 直接预测与真实价格走势。"""

    short_predictions = select_short_horizon_predictions(predictions, horizons=horizons)
    if short_predictions.empty:
        raise ValueError("没有可用于短期走势展示的预测记录")

    target_dates = sorted(pd.to_datetime(short_predictions["target_date"]).unique())
    display_start = target_dates[max(0, len(target_dates) - display_points)]
    short_predictions = short_predictions[
        pd.to_datetime(short_predictions["target_date"]) >= display_start
    ].copy()
    ordered_prices = prices.sort_values("date").reset_index(drop=True)
    actual = ordered_prices[
        (ordered_prices["date"] >= display_start)
        & (ordered_prices["date"] <= pd.to_datetime(short_predictions["target_date"]).max())
    ]

    colors = {1: "#2563eb", 3: "#059669", 5: "#d97706"}
    models = [model for model in ["Naive", "ARIMA", "XGBoost", "LSTM"] if model in set(short_predictions["model"])]
    figure, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex=True, sharey=True)
    for axis, model in zip(axes.flat, models):
        axis.plot(
            actual["date"],
            actual["close"],
            color="#111827",
            linewidth=2.0,
            label="Actual",
        )
        model_predictions = short_predictions[short_predictions["model"] == model]
        for horizon in horizons:
            series = model_predictions[model_predictions["horizon"] == horizon].sort_values("target_date")
            axis.plot(
                series["target_date"],
                series["predicted_close"],
                color=colors[horizon],
                linewidth=1.5,
                alpha=0.9,
                label=f"t+{horizon}",
            )
        axis.set_title(model)
        axis.grid(alpha=0.25)

    for axis in axes[:, 0]:
        axis.set_ylabel("Close Price")
    for axis in axes[-1, :]:
        axis.set_xlabel("Target Date")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle("Short-Horizon Direct Forecast Trajectories", y=0.99)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_short_horizon_mape(
    metrics: pd.DataFrame,
    output_path: Path,
    horizons: tuple[int, ...] = DEFAULT_SHORT_HORIZONS,
) -> None:
    """绘制 t+1、t+3、t+5 的模型 MAPE 折线对比。"""

    short_metrics = select_short_horizon_metrics(metrics, horizons=horizons)
    if short_metrics.empty:
        raise ValueError("没有可用于短期 MAPE 图的指标数据")

    colors = {"Naive": "#64748b", "ARIMA": "#2563eb", "XGBoost": "#059669", "LSTM": "#dc2626"}
    models = [model for model in ["Naive", "ARIMA", "XGBoost", "LSTM"] if model in set(short_metrics["model"])]
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for model in models:
        model_metrics = (
            short_metrics[short_metrics["model"] == model]
            .set_index("horizon")
            .reindex(horizons)
        )
        axis.plot(
            horizons,
            model_metrics["MAPE"].to_numpy(dtype=float) * 100,
            marker="o",
            markersize=6,
            linewidth=2.0,
            color=colors[model],
            label=model,
        )
    axis.set_title("Short-Horizon Forecast Error (MAPE)")
    axis.set_xlabel("Forecast Horizon (trading days)")
    axis.set_ylabel("MAPE (%)")
    axis.set_xticks(horizons, [f"t+{horizon}" for horizon in horizons])
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_train_validation_split(
    prices: pd.DataFrame,
    train_end_date: pd.Timestamp,
    validation_start_date: pd.Timestamp,
    output_path: Path,
) -> None:
    """绘制训练集和验证集划分图。"""

    plt.figure(figsize=(13, 5))
    plt.plot(prices["date"], prices["close"], color="#111827", linewidth=1.4, label="Close")
    plt.axvspan(
        validation_start_date,
        prices["date"].iloc[-1],
        color="#fee2e2",
        alpha=0.65,
        label="Validation",
    )
    plt.axvline(train_end_date, color="#dc2626", linestyle="--", linewidth=1.2, label="Train/Validation split")
    plt.title("Gold Price Train / Validation Split")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def run_prediction_experiment(
    db_path: str | Path = DEFAULT_DB_PATH,
    validation_ratio: float = 0.2,
    lstm_epochs: int = 60,
    arima_validation_limit: int | None = 260,
    run_multi_horizon: bool = False,
    multi_horizon_points: int = 120,
) -> dict[str, Path]:
    """运行完整预测实验，并返回输出文件路径。"""

    set_random_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prediction_data(db_path)
    train, validation = split_train_validation(prices, validation_ratio=validation_ratio)

    # ARIMA 滚动拟合较慢；默认取验证集最后约一年用于 PPT 对比。
    if arima_validation_limit and len(validation) > arima_validation_limit:
        validation_eval = validation.tail(arima_validation_limit).reset_index(drop=True)
        train_eval = prices[prices["date"] < validation_eval["date"].iloc[0]].reset_index(drop=True)
    else:
        train_eval = train
        validation_eval = validation

    feature_data = build_supervised_features(pd.concat([train_eval, validation_eval], ignore_index=True))
    train_end_date = pd.to_datetime(train_eval["date"].iloc[-1])
    feature_validation = feature_data[feature_data["date"] > train_end_date].reset_index(drop=True)

    results = pd.DataFrame(
        {
            "date": feature_validation["date"],
            "actual_close": feature_validation["target_close"],
        }
    )

    aligned_validation = validation_eval[
        validation_eval["date"].isin(set(results["date"]))
    ].reset_index(drop=True)
    aligned_train = prices[prices["date"] < results["date"].iloc[0]].reset_index(drop=True)

    results["Naive"] = naive_forecast(aligned_train, aligned_validation)
    results["ARIMA"] = arima_forecast(aligned_train, aligned_validation)
    results["XGBoost"] = xgboost_forecast(feature_data, train_end_date=train_end_date)
    results["LSTM"] = lstm_forecast(aligned_train, aligned_validation, epochs=lstm_epochs)

    metrics_rows = []
    for model_name in ["Naive", "ARIMA", "XGBoost", "LSTM"]:
        metrics = calculate_metrics(results["actual_close"].to_numpy(), results[model_name].to_numpy())
        metrics_rows.append(
            {
                "model": model_name,
                **metrics,
                "train_start": train_eval["date"].iloc[0].strftime("%Y-%m-%d"),
                "train_end": aligned_train["date"].iloc[-1].strftime("%Y-%m-%d"),
                "validation_start": results["date"].iloc[0].strftime("%Y-%m-%d"),
                "validation_end": results["date"].iloc[-1].strftime("%Y-%m-%d"),
                "validation_days": len(results),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    results_path = OUTPUT_DIR / "prediction_results.csv"
    metrics_path = OUTPUT_DIR / "prediction_metrics.csv"
    comparison_path = OUTPUT_DIR / "prediction_comparison.png"
    error_path = OUTPUT_DIR / "prediction_error.png"
    metrics_bar_path = OUTPUT_DIR / "prediction_metrics_bar.png"
    split_path = OUTPUT_DIR / "train_validation_split.png"

    results.to_csv(results_path, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    plot_prediction_comparison(results, comparison_path)
    plot_prediction_error(results, error_path)
    plot_metrics_bar(metrics_df, metrics_bar_path)
    plot_train_validation_split(
        prices,
        train_end_date=aligned_train["date"].iloc[-1],
        validation_start_date=results["date"].iloc[0],
        output_path=split_path,
    )

    outputs = {
        "results": results_path,
        "metrics": metrics_path,
        "comparison": comparison_path,
        "error": error_path,
        "metrics_bar": metrics_bar_path,
        "split": split_path,
    }
    if run_multi_horizon:
        multi_predictions, multi_metrics = run_multi_horizon_experiment(
            db_path=db_path,
            evaluation_points=multi_horizon_points,
            lstm_epochs=lstm_epochs,
        )
        multi_predictions_path = OUTPUT_DIR / "multi_horizon_predictions.csv"
        multi_metrics_path = OUTPUT_DIR / "multi_horizon_metrics.csv"
        multi_metrics_plot_path = OUTPUT_DIR / "multi_horizon_metrics.png"
        multi_example_path = OUTPUT_DIR / "multi_horizon_example.png"
        short_horizon_trajectory_path = OUTPUT_DIR / "short_horizon_trajectories.png"
        short_horizon_mape_path = OUTPUT_DIR / "short_horizon_mape.png"
        multi_predictions.to_csv(multi_predictions_path, index=False, encoding="utf-8-sig")
        multi_metrics.to_csv(multi_metrics_path, index=False, encoding="utf-8-sig")
        plot_multi_horizon_metrics(multi_metrics, multi_metrics_plot_path)
        example_origin = select_common_example_origin(multi_predictions)
        plot_multi_horizon_example(prices, multi_predictions, example_origin, multi_example_path)
        plot_short_horizon_trajectories(prices, multi_predictions, short_horizon_trajectory_path)
        plot_short_horizon_mape(multi_metrics, short_horizon_mape_path)
        outputs.update(
            {
                "multi_horizon_predictions": multi_predictions_path,
                "multi_horizon_metrics": multi_metrics_path,
                "multi_horizon_metrics_plot": multi_metrics_plot_path,
                "multi_horizon_example": multi_example_path,
                "short_horizon_trajectories": short_horizon_trajectory_path,
                "short_horizon_mape": short_horizon_mape_path,
            }
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="运行金价离线预测实验")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite 数据库路径")
    parser.add_argument("--validation-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--lstm-epochs", type=int, default=60, help="LSTM 训练轮数")
    parser.add_argument(
        "--arima-validation-limit",
        type=int,
        default=260,
        help="ARIMA 默认只回测验证集最后 N 个交易日；设为 0 表示使用完整验证集",
    )
    parser.add_argument(
        "--multi-horizon",
        action="store_true",
        help="额外运行 1、5、20、60 个交易日的直接多步长预测回测",
    )
    parser.add_argument(
        "--multi-horizon-points",
        type=int,
        default=120,
        help="多预测步长回测的每个步长验证预测起点数量",
    )
    args = parser.parse_args()

    outputs = run_prediction_experiment(
        db_path=args.db_path,
        validation_ratio=args.validation_ratio,
        lstm_epochs=args.lstm_epochs,
        arima_validation_limit=args.arima_validation_limit or None,
        run_multi_horizon=args.multi_horizon,
        multi_horizon_points=args.multi_horizon_points,
    )
    print("Prediction experiment finished.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
