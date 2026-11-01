"""Feature engineering.

Rule: only use information available **when the order is placed**. Anything
measured after (handling time, transit time, review) would leak the answer.
"""
import numpy as np
import pandas as pd

CATEGORICAL = [
    "customer_state",
    "seller_state",
    "customer_region",
    "main_category",
    "main_payment_type",
]
NUMERIC = [
    "promised_days",
    "seller_customer_distance_km",
    "is_same_state",
    "items_count",
    "sellers_count",
    "items_value",
    "freight_value",
    "freight_ratio",
    "total_weight_g",
    "total_volume_cm3",
    "max_installments",
    "purchase_month",
    "purchase_weekday",
    "purchase_hour",
    "seller_past_orders",
    "seller_past_late_rate",
    "state_past_late_rate",
    "route_past_actual_days",
    "promise_slack_days",
    "seller_past_handling_days",
]
TARGET = "is_late"


def _past_rate(df: pd.DataFrame, key: str, prior_n: int = 20) -> tuple[pd.Series, pd.Series]:
    """Late rate of `key` computed only on orders delivered **before** each purchase.

    An order's own outcome is unknown at purchase time, and so are the outcomes
    of orders still in transit. We therefore count, for each order, the orders
    of the same key whose delivery date is earlier than this purchase date.
    The rate is shrunk toward the global rate (Bayesian smoothing) so that a
    seller with 2 orders does not get a 0 % or 100 % score.
    """
    events = df[[key, "delivered_at", "is_late"]].rename(columns={"delivered_at": "t"})
    events = events.sort_values("t")
    events["cum_n"] = events.groupby(key).cumcount() + 1
    events["cum_late"] = events.groupby(key)["is_late"].cumsum()

    queries = df[[key, "purchased_at"]].reset_index().rename(columns={"purchased_at": "t"})
    queries = queries.sort_values("t")
    merged = pd.merge_asof(
        queries, events[[key, "t", "cum_n", "cum_late"]],
        on="t", by=key, direction="backward", allow_exact_matches=False,
    ).set_index("index").sort_index()

    n = merged["cum_n"].fillna(0)
    late = merged["cum_late"].fillna(0)
    global_rate = df["is_late"].mean()
    rate = (late + prior_n * global_rate) / (n + prior_n)
    return n, rate


def _past_mean(df: pd.DataFrame, key, value: str, prior_n: int = 10) -> pd.Series:
    """Mean of `value` for the same key over orders delivered before the purchase."""
    keys = [key] if isinstance(key, str) else list(key)
    tmp = df.copy()
    tmp["_k"] = tmp[keys].astype(str).agg("|".join, axis=1)
    events = tmp[["_k", "delivered_at", value]].rename(columns={"delivered_at": "t"}).sort_values("t")
    events["cum_n"] = events.groupby("_k").cumcount() + 1
    events["cum_sum"] = events.groupby("_k")[value].cumsum()
    queries = tmp[["_k", "purchased_at"]].reset_index().rename(columns={"purchased_at": "t"}).sort_values("t")
    merged = pd.merge_asof(
        queries, events[["_k", "t", "cum_n", "cum_sum"]],
        on="t", by="_k", direction="backward", allow_exact_matches=False,
    ).set_index("index").sort_index()
    n, s = merged["cum_n"].fillna(0), merged["cum_sum"].fillna(0)
    prior = df[value].mean()
    return (s + prior_n * prior) / (n + prior_n)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_same_state"] = out["is_same_state"].astype(float)
    out["freight_ratio"] = out["freight_value"] / out["items_value"].replace(0, np.nan)
    out["purchase_month"] = out["purchased_at"].dt.month
    out["purchase_weekday"] = out["purchased_at"].dt.weekday
    out["purchase_hour"] = out["purchased_at"].dt.hour
    out["seller_past_orders"], out["seller_past_late_rate"] = _past_rate(out, "main_seller_id")
    _, out["state_past_late_rate"] = _past_rate(out, "customer_state", prior_n=200)
    # How long this route (seller state -> customer state) really took recently,
    # and how much margin the promise leaves compared to it.
    out["route_past_actual_days"] = _past_mean(out, ["seller_state", "customer_state"], "actual_days")
    out["promise_slack_days"] = out["promised_days"] - out["route_past_actual_days"]
    out["seller_past_handling_days"] = _past_mean(out, "main_seller_id", "seller_handling_days")
    for col in CATEGORICAL:
        out[col] = out[col].fillna("unknown").astype("category")
    return out
