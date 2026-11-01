"""The historical features must never use information from the future."""
import pandas as pd

from delay.features import _past_rate


def test_past_rate_ignores_orders_not_yet_delivered():
    df = pd.DataFrame({
        "seller": ["A", "A", "A"],
        "purchased_at": pd.to_datetime(["2018-01-01", "2018-01-02", "2018-01-10"]),
        "delivered_at": pd.to_datetime(["2018-01-05", "2018-01-20", "2018-01-15"]),
        "is_late": [1, 1, 0],
    })
    n, _ = _past_rate(df, "seller", prior_n=0)
    # order 1: nothing delivered before 01-01
    # order 2: order 1 not delivered yet on 01-02
    # order 3: only order 1 delivered before 01-10 (order 2 arrives on 01-20)
    assert list(n) == [0, 0, 1]


def test_past_rate_is_smoothed_toward_global_rate():
    df = pd.DataFrame({
        "seller": ["A", "A"],
        "purchased_at": pd.to_datetime(["2018-01-01", "2018-02-01"]),
        "delivered_at": pd.to_datetime(["2018-01-05", "2018-02-05"]),
        "is_late": [1, 0],
    })
    _, rate = _past_rate(df, "seller", prior_n=20)
    global_rate = 0.5
    assert rate.iloc[0] == global_rate            # no history -> global rate
    assert global_rate < rate.iloc[1] < 1         # one late order pulls it up, not to 100 %
