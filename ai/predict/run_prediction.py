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

    return {
        "results": results_path,
        "metrics": metrics_path,
        "comparison": comparison_path,
        "error": error_path,
        "metrics_bar": metrics_bar_path,
        "split": split_path,
    }


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
    args = parser.parse_args()

    outputs = run_prediction_experiment(
        db_path=args.db_path,
        validation_ratio=args.validation_ratio,
        lstm_epochs=args.lstm_epochs,
        arima_validation_limit=args.arima_validation_limit or None,
    )
    print("Prediction experiment finished.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
