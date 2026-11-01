"""Command line entry point: python -m delay.train"""
import json
from pathlib import Path

import joblib

from .data import load_orders
from .features import build_features
from .model import baseline, evaluate, gradient_boosting, time_split

ROOT = Path(__file__).resolve().parents[2]


def main() -> dict:
    df = build_features(load_orders())
    s = time_split(df)
    results = {}
    for name, model in {"logistic_regression": baseline(), "gradient_boosting": gradient_boosting()}.items():
        model.fit(s.X_train, s.y_train)
        proba = model.predict_proba(s.X_test)[:, 1]
        results[name] = evaluate(s.y_test, proba)
        print(name, {k: round(v, 3) for k, v in results[name].items()})
        if name == "gradient_boosting":
            (ROOT / "models").mkdir(exist_ok=True)
            joblib.dump(model, ROOT / "models" / "gradient_boosting.joblib")
    results["train_rows"], results["test_rows"] = len(s.X_train), len(s.X_test)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
