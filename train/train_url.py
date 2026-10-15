import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from app.url_features import extract_features

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL_OUT = ROOT / "models" / "url_model.joblib"
METRICS_OUT = ROOT / "eval" / "url_metrics.json"


def load_dataset(path: Path, sample: int | None) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["URL", "label"])
    df = df.drop_duplicates(subset="URL").dropna()
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42)
    # PhiUSIIL uses label 1 = legitimate, 0 = phishing.
    # We want phishing = 1 so that a higher score means riskier.
    df["y"] = 1 - df["label"].astype(int)
    print("Class counts (1 = phishing):", df["y"].value_counts().to_dict())
    return df


def build_features(urls: pd.Series) -> pd.DataFrame:
    start = time.time()
    rows = [extract_features(u) for u in urls]
    print(f"Extracted features for {len(rows)} URLs in {time.time() - start:.1f}s")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Train the URL phishing model")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--sample", type=int, default=None,
                        help="Use a random subset for a quick test run, e.g. 20000")
    args = parser.parse_args()

    df = load_dataset(args.data, args.sample)
    X = build_features(df["URL"])
    y = df["y"].values

    X_train, X_test, y_train, y_test, urls_train, urls_test = train_test_split(
        X, y, df["URL"].tolist(), test_size=0.2, stratify=y, random_state=42
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate on the held-out test set
    start = time.time()
    proba = model.predict_proba(X_test)[:, 1]
    latency_ms = (time.time() - start) / len(X_test) * 1000
    pred = (proba >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    metrics = {
        "test_size": int(len(y_test)),
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "avg_inference_ms_per_url": round(latency_ms, 3),
        "false_positive_examples": [],
        "false_negative_examples": [],
    }

    # Keep examples of mistakes for the evaluation report
    for url, true, p in zip(urls_test, y_test, proba):
        pred_label = int(p >= 0.5)
        if pred_label == 1 and true == 0 and len(metrics["false_positive_examples"]) < 20:
            metrics["false_positive_examples"].append({"url": url, "score": round(float(p), 3)})
        if pred_label == 0 and true == 1 and len(metrics["false_negative_examples"]) < 20:
            metrics["false_negative_examples"].append({"url": url, "score": round(float(p), 3)})

    print(json.dumps({k: v for k, v in metrics.items() if "examples" not in k}, indent=2))

    # Save model, feature order and metrics
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": list(X.columns)}, MODEL_OUT)
    METRICS_OUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved model to {MODEL_OUT}")
    print(f"Saved metrics to {METRICS_OUT}")


if __name__ == "__main__":
    main()