import argparse
import csv
import json
import time
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)

from app.scoring import analyze

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval" / "message_metrics.json"

# Hand-written labelled set. 1 = scam or phishing, 0 = legitimate.
# Includes some hard cases: no brand names, no obvious URL, and legitimate
# messages that use scam-like words (payment, password, net banking).
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
    # Modified or new attacks (no brand name, no obvious URL, or unusual wording)
    ("Unusual sign-in activity on your account. Please sign in to review: http://google-accounts-review.xyz/signin", 1),
    ("Hey, check this out: http://bit.ly/3xYz9", 1),
    ("Your package couldn't be delivered. Reschedule here: https://dhl-track-parcel.com/reschedule", 1),
    ("Final notice: your tax return has an error. Confirm your details today to avoid penalty.", 1),
    ("Part-time job offer: like posts and get paid daily. Join our telegram group to start.", 1),
    ("Your account will be closed. Reply with your password to keep it open.", 1),
]


def load_csv(path: Path) -> list[tuple[str, int]]:
    """Load your own labelled set. Columns: text, label (1/0 or phishing/legit)."""
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw = row["label"].strip().lower()
            label = 1 if raw in {"1", "phishing", "spam", "scam"} else 0
            rows.append((row["text"], label))
    return rows


def run(samples: list[tuple[str, int]]) -> dict:
    y_true, y_pred, details = [], [], []
    latencies = []

    for text, label in samples:
        start = time.perf_counter()
        result = analyze(text)
        latencies.append((time.perf_counter() - start) * 1000)

        # A message counts as flagged if it is Medium or High
        flagged = int(result["risk"] != "Low")
        y_true.append(label)
        y_pred.append(flagged)
        details.append({
            "text": text,
            "true_label": label,
            "risk": result["risk"],
            "score": result["score"],
            "reasons": result["reasons"],
        })

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "total": len(samples),
        "positives_scam": sum(y_true),
        "negatives_legit": len(y_true) - sum(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
        "false_positives": [d for d in details if d["true_label"] == 0 and d["risk"] != "Low"],
        "false_negatives": [d for d in details if d["true_label"] == 1 and d["risk"] == "Low"],
    }
    return metrics


def print_summary(m: dict) -> None:
    c = m["confusion_matrix"]
    print(f"\nSamples: {m['total']} ({m['positives_scam']} scam, {m['negatives_legit']} legitimate)")
    print(f"Accuracy:  {m['accuracy']}")
    print(f"Precision: {m['precision']}")
    print(f"Recall:    {m['recall']}")
    print(f"F1:        {m['f1']}")
    print(f"Confusion: TN={c['tn']} FP={c['fp']} FN={c['fn']} TP={c['tp']}")
    print(f"Avg latency: {m['avg_latency_ms']} ms per message")

    print(f"\nFalse positives ({len(m['false_positives'])}): legitimate messages flagged")
    for d in m["false_positives"]:
        print(f"  [{d['risk']} {d['score']}] {d['text']}")
        for r in d["reasons"][:3]:
            print(f"      - {r}")

    print(f"\nFalse negatives ({len(m['false_negatives'])}): scams missed")
    for d in m["false_negatives"]:
        print(f"  [{d['risk']} {d['score']}] {d['text']}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the message scorer")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Optional labelled CSV with columns: text, label")
    args = parser.parse_args()

    samples = load_csv(args.csv) if args.csv else LABELLED
    metrics = run(samples)
    print_summary(metrics)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()