import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd
import tldextract

from app.url_features import extract_features, url_reasons
from app.text_rules import text_score

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "url_model.joblib"
_ext = tldextract.TLDExtract(suffix_list_urls=())

# Weights for the combined score (from the PRD): ML 0.4, URL rules 0.3, text rules 0.2
W_ML, W_URL, W_TEXT = 0.4, 0.3, 0.2
LOW_MAX, MEDIUM_MAX = 0.35, 0.70

# A critical rule means the link is almost certainly malicious, so the message is at least High
HARD_FLOOR = 0.75
CRITICAL_MARKERS = (
    "looks like the brand",
    "uses an IP address",
    "punycode",
    "'@'",
)

# Well-known legitimate domains. Matched on the registered domain, so
# google.com.evil.xyz is NOT trusted (its registered domain is evil.xyz).
TRUSTED_DOMAINS = {
    "google.com", "youtube.com", "facebook.com", "instagram.com", "whatsapp.com",
    "github.com", "microsoft.com", "apple.com", "wikipedia.org", "bbc.com",
    "flipkart.com", "amazon.in", "amazon.com", "swiggy.com", "netflix.com",
    "paytm.com", "paypal.com", "sbi.co.in", "hdfcbank.com", "icicibank.com",
    "axisbank.com", "npci.org.in", "irs.gov",
}
TRUSTED_SUFFIXES = ("gov.in", "gov", "edu", "ac.in")

URL_RE = re.compile(r"(?:https?://)?(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s]*)?", re.IGNORECASE)


@lru_cache(maxsize=1)
def load_url_model():
    """Load the trained model once. Returns (model, feature_order)."""
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["features"]


def _host(url: str) -> str:
    u = url.strip().lower()
    if "://" not in u:
        u = "http://" + u
    return urlparse(u).hostname or ""


def is_trusted(url: str) -> bool:
    """True if the link's registered domain is on the trusted list."""
    ext = _ext(_host(url))
    if not ext.registered_domain:
        return False
    if ext.registered_domain in TRUSTED_DOMAINS:
        return True
    return ext.suffix.endswith(TRUSTED_SUFFIXES)


def extract_urls(text: str) -> list[str]:
    """Find URL-like strings in a message."""
    found = []
    for m in URL_RE.finditer(text):
        candidate = m.group(0).rstrip(".,;:)!?")
        if re.search(r"[a-z]{2,}$", candidate, re.IGNORECASE):
            found.append(candidate)
    return found


def ml_url_score(url: str) -> float:
    """Phishing probability (0-1) from the trained URL model."""
    model, features = load_url_model()
    row = pd.DataFrame([extract_features(url)])[features]
    return float(model.predict_proba(row)[0, 1])


def url_rule_score(reasons: list[str]) -> float:
    """Each URL rule hit adds 0.25, capped at 1.0."""
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
    feed_hits: URLs found in threat feeds (from threat_intel.py). Any hit gives a score of 1.0.
    """
    feed_hits = feed_hits or set()
    urls = extract_urls(text)

    reasons: list[str] = []
    critical: list[str] = []
    url_details = []
    ml_part = url_part = 0.0

    for url in urls:
        in_feed = url in feed_hits

        # Trusted domains skip the model and URL rules, unless the feed lists them
        if is_trusted(url) and not in_feed:
            url_details.append({"url": url, "ml_score": 0.0, "trusted_domain": True, "reasons": []})
            continue

        ml = ml_url_score(url)
        rules = url_reasons(url)
        ml_part = max(ml_part, ml)
        url_part = max(url_part, url_rule_score(rules))
        url_details.append({"url": url, "ml_score": round(ml, 3), "trusted_domain": False, "reasons": rules})

        for r in rules:
            reasons.append(f"{url}: {r}")
            if any(marker in r for marker in CRITICAL_MARKERS):
                critical.append(f"{url}: {r}")

        if in_feed:
            reasons.append(f"{url}: found in a known threat feed")
            critical.append(f"{url}: found in a known threat feed")

    txt_part, txt_reasons = text_score(text)
    reasons.extend(txt_reasons)

    # The model alone can't push a message above Low when no rule supports it
    if urls and not reasons:
        ml_part = min(ml_part, 0.5)

    if urls:
        score = W_ML * ml_part + W_URL * url_part + W_TEXT * txt_part
    else:
        score = txt_part

    if critical:
        score = max(score, HARD_FLOOR)

    feed_override = any(u in feed_hits for u in urls)
    if feed_override:
        score = 1.0

    score = round(min(score, 1.0), 3)

    return {
        "risk": risk_level(score),
        "score": score,
        "reasons": reasons,
        "critical_rules": critical,
        "urls": url_details,
        "text_score": round(txt_part, 3),
        "feed_override": feed_override,
    }


if __name__ == "__main__":
    samples = [
        "Check this out: https://google.com",
        "Your order has shipped. Track it at https://www.flipkart.com/orders",
        "URGENT: your SBI account is locked. Verify now at http://sbi-secure.xyz/login",
        "Pay the Rs 500 fine within 24 hours or face arrest: http://paypa1.com/pay",
    ]
    for s in samples:
        result = analyze(s)
        print(f"\n{s}\n -> {result['risk']} ({result['score']})")
        for r in result["reasons"]:
            print("    -", r)