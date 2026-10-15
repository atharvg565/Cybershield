import re
import ipaddress
from urllib.parse import urlparse

import tldextract
from rapidfuzz.distance import Levenshtein

# Offline mode: uses the suffix list bundled with tldextract, no network call
_extract = tldextract.TLDExtract(suffix_list_urls=())

BRANDS = [
    "paypal", "amazon", "google", "microsoft", "apple", "netflix", "facebook",
    "instagram", "whatsapp", "flipkart", "paytm", "sbi", "hdfc", "icici",
    "axisbank", "irs", "incometax", "npci", "uidai",
]
SUSPICIOUS_TLDS = {"xyz", "top", "tk", "ml", "ga", "cf", "gq", "zip", "click", "country", "link", "work"}
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "rb.gy", "cutt.ly"}
SENSITIVE_WORDS = ["login", "verify", "secure", "update", "account", "bank", "kyc", "refund", "unlock", "confirm"]


def _normalize(url: str) -> str:
    url = url.strip()
    if not re.match(r"^[a-z][a-z0-9+.-]*://", url, re.IGNORECASE):
        url = "http://" + url
    return url


def _lookalike_brand(domain: str):
    """Return the brand this domain imitates, if it is close but not identical."""
    # Undo common character swaps: paypa1 -> paypal, g00gle -> google
    folded = domain.translate(str.maketrans({"1": "l", "0": "o", "3": "e", "5": "s"}))
    for brand in BRANDS:
        if domain == brand:
            return None  # the real brand
        if folded == brand or Levenshtein.distance(folded, brand) <= 2:
            return brand
    return None


def extract_features(url: str) -> dict:
    """Numeric features for the ML model."""
    url = _normalize(url)
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    ext = _extract(host)
    registered = ext.registered_domain or host
    subdomains = [s for s in ext.subdomain.split(".") if s]
    full_lower = url.lower()

    try:
        ipaddress.ip_address(host)
        has_ip = 1
    except ValueError:
        has_ip = 0

    return {
        "url_length": len(url),
        "host_length": len(host),
        "num_dots": host.count("."),
        "num_hyphens": host.count("-"),
        "num_digits": sum(c.isdigit() for c in host),
        "num_subdomains": len(subdomains),
        "has_ip": has_ip,
        "has_at_symbol": int("@" in url),
        "has_punycode": int("xn--" in host),
        "uses_https": int(parsed.scheme == "https"),
        "suspicious_tld": int(ext.suffix.split(".")[-1] in SUSPICIOUS_TLDS),
        "is_shortener": int(registered in SHORTENERS),
        "sensitive_word_count": sum(w in full_lower for w in SENSITIVE_WORDS),
        "brand_in_subdomain": int(any(b in ".".join(subdomains) for b in BRANDS)),
        "lookalike_brand": int(_lookalike_brand(ext.domain) is not None),
        "path_length": len(parsed.path),
    }


def url_reasons(url: str) -> list[str]:
    """Human-readable indicators for the explanation panel."""
    url = _normalize(url)
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    ext = _extract(host)
    reasons = []

    f = extract_features(url)
    brand = _lookalike_brand(ext.domain)
    if f["has_ip"]:
        reasons.append("Link uses an IP address instead of a domain name")
    if f["has_at_symbol"]:
        reasons.append("URL contains '@', which can hide the real destination")
    if f["has_punycode"]:
        reasons.append("Domain uses punycode (xn--), which can disguise letters")
    if brand:
        reasons.append(f"Domain '{ext.registered_domain}' looks like the brand '{brand}'")
    if f["brand_in_subdomain"] and not brand:
        reasons.append("A brand name appears in the subdomain but not in the registered domain")
    if f["is_shortener"]:
        reasons.append("Link is a URL shortener, so the true destination is hidden")
    if f["suspicious_tld"]:
        reasons.append(f"Uses a high-risk domain extension (.{ext.suffix.split('.')[-1]})")
    if not f["uses_https"]:
        reasons.append("Connection is not HTTPS")
    if f["num_subdomains"] >= 3:
        reasons.append(f"Unusually many subdomains ({f['num_subdomains']})")
    if f["sensitive_word_count"]:
        reasons.append("Contains credential-related words such as login, verify or secure")
    return reasons


if __name__ == "__main__":
    tests = [
        "https://www.sbi.co.in",
        "http://sbi-secure.xyz/login",
        "http://paypa1.com/verify",
        "http://192.168.4.7/bank",
        "bit.ly/3abc",
    ]
    for t in tests:
        print(t, "->", url_reasons(t))