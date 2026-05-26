import pandas as pd

from analysis.statistics import calculate_return_statistics


def test_calculate_return_statistics():
    df = pd.DataFrame(
        {
            "date": ["2026-05-24", "2026-05-25", "2026-05-26"],
            "close": [100.0, 110.0, 121.0],
        }
    )

    stats = calculate_return_statistics(df)

    assert stats["count"] == 2.0
    assert round(stats["mean_return"], 6) == 0.1
    assert round(stats["std_return"], 6) == 0.0
    assert round(stats["max_return"], 6) == 0.1
    assert round(stats["min_return"], 6) == 0.1
