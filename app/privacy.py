import hashlib
import re

# Order matters: card numbers are checked before phone numbers,
# because a long digit string could match either pattern.
PATTERNS = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("CARD", re.compile(r"(?<![\d.])\d(?:[ -]?\d){12,18}(?![\d.])")),
    ("AADHAAR", re.compile(r"(?<![\d.])\d{4}[ -]?\d{4}[ -]?\d{4}(?![\d.])")),
    ("PHONE", re.compile(r"(?<![\d.])(?:\+?\d{1,3}[ -]?)?\d{5}[ -]?\d{5}(?![\d.])")),
]


def mask_pii(text: str) -> tuple[str, dict]:
    """
    Replace personal data with placeholders before analysis.
    Returns (masked_text, counts), e.g. ("Call [PHONE]", {"PHONE": 1}).
    URLs are left intact so the link checks still work.
    """
    counts: dict[str, int] = {}
    masked = text
    for label, pattern in PATTERNS:
        masked, n = pattern.subn(f"[{label}]", masked)
        if n:
            counts[label] = n
    return masked, counts


def fingerprint(text: str) -> str:
    """
    One-way hash of a message. Store this instead of the raw text,
    so you can spot repeat submissions without keeping personal data.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    samples = [
        "Call me on +91 98765 43210 or mail ravi.k@gmail.com",
        "Your card 4111 1111 1111 1111 will be charged. Aadhaar 1234 5678 9012 is linked.",
        "Track your order at https://www.flipkart.com/orders",
    ]
    for s in samples:
        masked, counts = mask_pii(s)
        print(f"\nIn:     {s}")
        print(f"Masked: {masked}")
        print(f"Found:  {counts}")
        print(f"Hash:   {fingerprint(s)[:16]}...")