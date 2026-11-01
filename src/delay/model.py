"""Train and evaluate the late-delivery classifier with a time-based split."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL, NUMERIC, TARGET

# Train on orders placed before this date, test on the following months.
# A random split would let the model "see the future" (same weeks, same sellers).
SPLIT_DATE = "2018-03-01"


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def time_split(df: pd.DataFrame) -> Split:
    train = df[df["purchased_at"] < SPLIT_DATE]
    test = df[df["purchased_at"] >= SPLIT_DATE]
    cols = CATEGORICAL + NUMERIC
    return Split(train[cols], test[cols], train[TARGET], test[TARGET])


def baseline() -> object:
    """Logistic regression: the 'simple and explainable' reference."""
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CATEGORICAL),
    ])
    return make_pipeline(pre, LogisticRegression(max_iter=2000, class_weight="balanced"))


def gradient_boosting() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype",
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=80,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )


def evaluate(y_true: pd.Series, proba: np.ndarray) -> dict:
    """Ranking metrics + the business view: how many late orders do we catch
    if the ops team can only check the 10 % riskiest orders?"""
    top = np.argsort(-proba)[: max(1, int(0.10 * len(proba)))]
    caught = y_true.iloc[top].sum()
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "base_rate": y_true.mean(),
        "precision_top10": y_true.iloc[top].mean(),
        "recall_top10": caught / y_true.sum(),
        "lift_top10": y_true.iloc[top].mean() / y_true.mean(),
    }
