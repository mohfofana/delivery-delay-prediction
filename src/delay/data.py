"""Load the order fact table produced by the ecommerce-data-pipeline project."""
from pathlib import Path

import pandas as pd

PIPELINE_URL = (
    "https://raw.githubusercontent.com/mohfofana/ecommerce-data-pipeline/"
    "main/exports/fct_orders.parquet"
)
LOCAL = Path(__file__).resolve().parents[2] / "data" / "fct_orders.parquet"


def load_orders() -> pd.DataFrame:
    """Return delivered orders with a known delivery date.

    Uses the local copy in data/ if present, otherwise downloads the mart
    exported by the pipeline repository.
    """
    if not LOCAL.exists():
        LOCAL.parent.mkdir(exist_ok=True)
        pd.read_parquet(PIPELINE_URL).to_parquet(LOCAL)
    df = pd.read_parquet(LOCAL)
    df = df[(df["order_status"] == "delivered") & df["delivered_at"].notna()].copy()
    df["is_late"] = df["is_late"].astype(int)
    return df.sort_values("purchased_at").reset_index(drop=True)
