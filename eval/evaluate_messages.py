import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

from app.scoring import analyze

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval" / "message_metrics.json"

LABELLED = [
    # Legitimate
    ("Your order has shipped. Track it at https://www.flipkart.com/orders", 0),
    ("Your Amazon package will arrive tomorrow. Track: https://www.amazon.in/gp/your-account/order-history", 0),
    ("Hi, are we still meeting for lunch at 1pm?", 0),
    ("Your electricity bill for September is ready. View it in the app.", 0),
    ("Reminder: your dentist appointment is on Monday at 10am.", 0),
    ("Team, the quarterly report is attached. Please review before Friday.", 0),
    ("Your payment of Rs 499 to Netflix was successful.", 0),
    ("Happy birthday! Hope you have a great day.", 0),
    ("Your flight AI101 is confirmed. Check-in opens 24 hours before departure.", 0),
    ("Wikipedia article on photosynthesis: https://en.wikipedia.org/wiki/Photosynthesis", 0),
    ("Your password was changed successfully. If this was not you, contact support.", 0),
    ("Thanks for subscribing. Read the latest news at https://www.bbc.com/news", 0),
    ("The meeting notes are in the shared drive. See you tomorrow.", 0),
    ("Your Swiggy order is out for delivery.", 0),
    ("Mom: call me when you get a chance.", 0),
    ("Your HDFC statement for August is available on the net banking portal.", 0),
    ("Your parcel was delivered today. Thank you for shopping with us.", 0),
    ("Your Google Calendar event starts in 15 minutes.", 0),
    ("Check this out: https://google.com", 0),

    # Scam or phishing
    ("URGENT: your SBI account is locked. Verify now at http://sbi-secure.xyz/login", 1),
    ("Pay the Rs 500 fine within 24 hours or face arrest: http://paypa1.com/pay", 1),
    ("Congratulations! You won a lottery. Pay a processing fee to claim it.", 1),
    ("Your parcel is held at customs. Pay Rs 80 fee: http://dhl-delivery-track.top/pay", 1),
    ("Work from home and earn 5000 per day. Message us on WhatsApp to start.", 1),
    ("Guaranteed returns! Invest now in our trading tips group.", 1),
    ("Your Netflix account will be suspended. Update your payment details: http://netflix-billing-help.click", 1),
    ("Your KYC is pending. Update your details within 24 hours to avoid account block: http://sbi-kyc-update.xyz", 1),
    ("Income tax refund of Rs 12,400 pending. Claim here: http://incometax-refund.work", 1),
    ("Send bitcoin to the wallet below to unlock your funds immediately.", 1),
    ("Dear customer, your debit card will be blocked. Share your card number and CVV to reactivate.", 1),
    ("Your PAN card is not linked. Verify now or face penalty: http://pan-link-verify.top", 1),
    ("Hi, it's me, I lost my phone. Send me money on this UPI id urgently.", 1),
    ("Your WhatsApp will be deactivated. Confirm identity: http://192.168.4.7/whatsapp", 1),
    ("Unusual sign-in activity on your account. Please sign in to review: http://google-accounts-review.xyz/signin", 1),
    ("Hey, check this out: http://bit.ly/3xYz9", 1),
    ("Your package couldn't be delivered. Reschedule here: https://dhl-track-parcel.com/reschedule", 1),
    ("Final notice: your tax return has an error. Confirm your details today to avoid penalty.", 1),
    ("Part-time job offer: like posts and get paid daily. Join our telegram group to start.", 1),
    ("Your account will be closed. Reply with your password to keep it open.", 1),
    ("20 million credited to your account. Pay 1 million to be able to withdraw.", 1),
]


def load_csv(path: Path) -> list[tuple[str, int]]:
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw = row["label"].strip().lower()
            label = 1 if raw in {"1", "phishing", "spam", "scam"} else 0
            rows.append((row["text"], label))
    return rows


def safe_div(a, b):
    return round(a / b, 4) if b else 0.0


def run(samples):
    y_true, scores, details, latencies = [], [], [], []
    for text, label in samples:
        start = time.perf_counter()
        result = analyze(text)
        latencies.append((time.perf_counter() - start) * 1000)
        y_true.append(label)
        scores.append(result["score"])
        details.append({
            "text": text, "true_label": label, "risk": result["risk"],
            "score": result["score"], "reasons": result["reasons"],
        })

    y_true = np.array(y_true)
    scores = np.array(scores)
    y_pred = (scores > 0.35).astype(int)  # flagged = Medium or High

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    n_pos, n_neg = int(tp + fn), int(tn + fp)

    # Rule hit counts: how often each reason fires on scams vs legitimate messages
    rule_hits: dict[str, dict[str, int]] = {}
    for d in details:
        for r in d["reasons"]:
            key = r.split(': "')[0].split(": ")[-1][:90]
            bucket = rule_hits.setdefault(key, {"scam": 0, "legit": 0})
            bucket["scam" if d["true_label"] == 1 else "legit"] += 1

    # Threshold sweep for ROC and precision-recall curves
    thresholds = [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sweep = []
    for t in thresholds:
        p = (scores >= t).astype(int) if t > 0 else np.ones_like(y_true)
        ctn, cfp, cfn, ctp = confusion_matrix(y_true, p, labels=[0, 1]).ravel()
        sweep.append({
            "threshold": t,
            "tpr_recall": safe_div(ctp, ctp + cfn),
            "fpr": safe_div(cfp, cfp + ctn),
            "precision": safe_div(ctp, ctp + cfp),
        })

    roc_auc = round(float(roc_auc_score(y_true, scores)), 4) if len(set(y_true)) > 1 else None
    pr_auc = round(float(average_precision_score(y_true, scores)), 4) if len(set(y_true)) > 1 else None

    metrics = {
        "total": n,
        "positives_scam": n_pos,
        "negatives_legit": n_neg,
        "decision_threshold": 0.35,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "confusion_matrix_rates": {
            "true_negative_rate": safe_div(tn, n_neg),
            "false_positive_rate": safe_div(fp, n_neg),
            "false_negative_rate": safe_div(fn, n_pos),
            "true_positive_rate": safe_div(tp, n_pos),
        },
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "specificity": safe_div(tn, n_neg),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "matthews_corrcoef": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold_sweep": sweep,
        "rule_hits": rule_hits,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
        "false_positives": [d for d in details if d["true_label"] == 0 and d["risk"] != "Low"],
        "false_negatives": [d for d in details if d["true_label"] == 1 and d["risk"] == "Low"],
    }
    return metrics


def print_summary(m):
    c = m["confusion_matrix"]
    r = m["confusion_matrix_rates"]
    print(f"\nSamples: {m['total']} ({m['positives_scam']} scam, {m['negatives_legit']} legitimate)")
    print(f"Confusion: TN={c['tn']} FP={c['fp']} FN={c['fn']} TP={c['tp']}")
    print(f"Rates: TNR={r['true_negative_rate']} FPR={r['false_positive_rate']} "
          f"FNR={r['false_negative_rate']} TPR={r['true_positive_rate']}")
    print(f"Accuracy={m['accuracy']} Precision={m['precision']} Recall={m['recall']} "
          f"Specificity={m['specificity']} F1={m['f1']} MCC={m['matthews_corrcoef']}")
    print(f"ROC AUC={m['roc_auc']} PR AUC={m['pr_auc']} Avg latency={m['avg_latency_ms']} ms")
    print(f"\nFalse positives ({len(m['false_positives'])}):")
    for d in m["false_positives"]:
        print(f"  [{d['risk']} {d['score']}] {d['text']}")
    print(f"\nFalse negatives ({len(m['false_negatives'])}):")
    for d in m["false_negatives"]:
        print(f"  [{d['risk']} {d['score']}] {d['text']}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the message scorer")
    parser.add_argument("--csv", type=Path, default=None, help="Optional labelled CSV: text, label")
    args = parser.parse_args()

    samples = load_csv(args.csv) if args.csv else LABELLED
    metrics = run(samples)
    print_summary(metrics)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()