import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_OUT = ROOT / "models" / "text_model.joblib"
METRICS_OUT = ROOT / "eval" / "text_metrics.json"

LABEL_COLS = ["v1", "label", "category", "class", "type"]
TEXT_COLS = ["v2", "text", "message", "sms", "body"]


def find_data_file() -> Path:
    for name in ["sms_spam.csv", "spam.csv", "SMSSpamCollection", "SMSSpamCollection.txt"]:
        p = DATA_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError("Put the SMS dataset in the data folder as sms_spam.csv")


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1", sep=None, engine="python")
    cols = {c.strip().lower(): c for c in df.columns}
    label_col = next((cols[c] for c in LABEL_COLS if c in cols), None)
    text_col = next((cols[c] for c in TEXT_COLS if c in cols), None)

    if label_col is None or text_col is None:
        # Raw UCI file: tab separated, no header
        df = pd.read_csv(path, encoding="latin-1", sep="\t", header=None, names=["label", "text"])
        label_col, text_col = "label", "text"

    df = df[[label_col, text_col]].dropna()
    df.columns = ["label", "text"]
    df["y"] = df["label"].astype(str).str.strip().str.lower().map(
        {"spam": 1, "ham": 0, "1": 1, "0": 0, "phishing": 1, "legit": 0}
    )
    df = df.dropna(subset=["y"]).drop_duplicates(subset="text")
    df["y"] = df["y"].astype(int)
    return df


def main():
    parser = argparse.ArgumentParser(description="Train the text scam model")
    parser.add_argument("--data", type=Path, default=None)
    args = parser.parse_args()

    path = args.data or find_data_file()
    df = load_data(path)
    print(f"Loaded {len(df)} messages from {path.name}: {df['y'].value_counts().to_dict()} (1 = spam)")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["y"], test_size=0.2, stratify=df["y"], random_state=42
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    metrics = {
        "test_size": int(len(y_test)),
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    # Words the model treats as strongest scam signals (used for explanations)
    vocab = pipe.named_steps["tfidf"].get_feature_names_out()
    coefs = pipe.named_steps["clf"].coef_[0]
    top = sorted(zip(vocab, coefs), key=lambda x: -x[1])[:25]
    metrics["top_scam_terms"] = [{"term": t, "weight": round(float(w), 3)} for t, w in top]

    print(json.dumps({k: v for k, v in metrics.items() if k != "top_scam_terms"}, indent=2))
    print("Top scam terms:", ", ".join(t for t, _ in top[:12]))

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_OUT)
    METRICS_OUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved model to {MODEL_OUT}")


if __name__ == "__main__":
    main()