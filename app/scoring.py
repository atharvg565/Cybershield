import re
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from app.url_features import extract_features, url_reasons
from app.text_rules import text_score

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "url_model.joblib"

# PRD weights: 0.4 ML + 0.3 URL rules + 0.2 text rules, with a feed override
W_ML, W_URL, W_TEXT = 0.4, 0.3, 0.2

LOW_MAX = 0.35
MEDIUM_MAX = 0.70

URL_RE = re.compile(r"(?:https?://)?(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s]*)?", re.IGNORECASE)


@lru_cache(maxsize=1)
def load_url_model():
    """Load the trained model once. Returns (model, feature_order)."""
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["features"]


def extract_urls(text: str) -> list[str]:
    """Find URL-like strings in a message. Skips bare words that just contain a dot."""
    found = []
    for m in URL_RE.finditer(text):
        candidate = m.group(0).rstrip(".,;:)!?")
        # Ignore things like "e.g" or "3.5" that look like domains
        if re.search(r"[a-z]{2,}$", candidate, re.IGNORECASE):
            found.append(candidate)
    return found


def ml_url_score(url: str) -> float:
    """Phishing probability (0-1) from the trained URL model."""
    model, features = load_url_model()
    row = pd.DataFrame([extract_features(url)])[features]
    return float(model.predict_proba(row)[0, 1])


def url_rule_score(reasons: list[str]) -> float:
    """Each rule reason adds weight, capped at 1.0."""
    return min(1.0, 0.25 * len(reasons))


def risk_level(score: float) -> str:
    if score < LOW_MAX:
        return "Low"
    if score <= MEDIUM_MAX:
        return "Medium"
    return "High"


def analyze(text: str, feed_hits: set[str] | None = None) -> dict:
    """
    Analyse an email, SMS or URL and return a risk verdict with reasons.
    feed_hits: URLs found in threat feeds (from threat_intel.py). Any hit is an instant High.
    """
    feed_hits = feed_hits or set()
    urls = extract_urls(text)

    reasons: list[str] = []
    url_details = []
    ml_scores, rule_scores = [], []

    for url in urls:
        ml = ml_url_score(url)
        rules = url_reasons(url)
        ml_scores.append(ml)
        rule_scores.append(url_rule_score(rules))
        url_details.append({"url": url, "ml_score": round(ml, 3), "reasons": rules})
        reasons.extend(f"{url}: {r}" for r in rules)
        if url in feed_hits:
            reasons.append(f"{url}: found in a known threat feed")

    # Use the worst URL, since one bad link is enough to make the message dangerous
    ml_part = max(ml_scores, default=0.0)
    url_part = max(rule_scores, default=0.0)

    txt_part, txt_reasons = text_score(text)
    reasons.extend(txt_reasons)

    # ML alone should not raise a message above Low without a supporting rule hit
    if not reasons:
        ml_part = min(ml_part, 0.5)

    # A lookalike brand domain is a strong signal on its own
    lookalike = any("looks like the brand" in r for r in reasons)

    # If the message has no URLs, the URL weights are redistributed to the text rules
    if urls:
        score = W_ML * ml_part + W_URL * url_part + W_TEXT * txt_part
    else:
        score = txt_part

    # A lookalike domain floors the score at High
    if lookalike:
        score = max(score, 0.75)

    feed_override = any(u in feed_hits for u in urls)
    if feed_override:
        score = 1.0

    score = round(min(score, 1.0), 3)

    return {
        "risk": risk_level(score),
        "score": score,
        "reasons": reasons,
        "urls": url_details,
        "text_score": round(txt_part, 3),
        "feed_override": feed_override,
    }


if __name__ == "__main__":
    samples = [
        "Your order has shipped. Track it at https://www.flipkart.com/orders",
        "URGENT: your SBI account is locked. Verify now at http://sbi-secure.xyz/login",
        "Pay the Rs 500 fine within 24 hours or face arrest: http://paypa1.com/pay",
    ]
    for s in samples:
        result = analyze(s)
        print(f"\n{s}\n -> {result['risk']} ({result['score']})")
        for r in result["reasons"]:
            print("    -", r)