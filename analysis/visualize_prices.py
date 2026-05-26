"""Price visualization helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_close_line(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Generate a simple close-price line chart."""

    if df.empty:
        raise ValueError("Cannot plot an empty price dataframe.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    working = df.copy()
    working["date"] = pd.to_datetime(working["date"])
    working = working.sort_values("date")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(working["date"], working["close"], label="Close", linewidth=1.8)
    ax.set_title("Gold Close Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)

    return output
