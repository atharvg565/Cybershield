import re

# Each rule: (regex, weight, reason shown to the user)
RULES = [
    # Urgency and threats
    (r"\b(urgent|immediately|within \d+ (hours?|minutes?)|last warning|final notice|act now)\b",
     0.15, "Uses urgency or time pressure to rush you"),
    (r"\b(account (is )?(locked|suspended|blocked|closed|disabled)|will be (blocked|suspended|closed))\b",
     0.20, "Threatens to lock or close your account"),
    (r"\b(arrest|legal action|court|police|warrant|penalty|fine)\b",
     0.15, "Uses fear of legal trouble or penalties"),
    (r"\b(pay|fine|fee)\b.*\b(within \d+ hours?|or face|or be|arrest)\b",
     0.25, "Demands payment under threat of penalty or arrest"),

    # Credential and sensitive-data requests
    (r"\b(verify|confirm|update|validate) (your )?(account|identity|details|kyc|information|password|login)\b",
     0.20, "Asks you to verify or update account details"),
    (r"\b(password|otp|one[- ]time (password|code)|pin|cvv|card number|net ?banking|login details)\b",
     0.25, "Asks for passwords, OTPs or card details"),
    (r"\b(aadhaar|aadhar|pan card|ssn|social security)\b",
     0.20, "Asks for identity numbers"),

    # Payment demands
    (r"\b(pay|payment|transfer|send money|deposit|processing fee|registration fee|refund|fee)\b",
     0.10, "Mentions a payment or money transfer"),
    (r"\b(gift card|bitcoin|crypto|usdt|wire transfer)\b",
     0.20, "Requests gift cards or crypto, a common scam method"),

    # Impersonation of banks, government and brands
    (r"\b(sbi|hdfc|icici|axis bank|paytm|paypal|amazon|netflix|income tax|irs|uidai|npci|rbi|government)\b",
     0.10, "Claims to be a bank, government body or well-known brand"),

    # Scam themes
    (r"\b(lottery|won a prize|congratulations you|cashback offer|work from home|earn \d+ ?(per|a) (day|week)|part[- ]time job)\b",
     0.15, "Promises prizes, easy money or a job offer"),
    (r"\b(won|prize|lottery)\b.*\b(fee|pay|claim)\b",
     0.30, "Prize offer that requires a fee to claim"),
    (r"\b(investment|guaranteed returns|double your money|trading tips)\b",
     0.15, "Promises investment returns"),
    (r"\b(whatsapp|telegram)\b.*\b(job|task|pay|hire|recruit)\b",
     0.15, "Asks you to continue a job or payment on a messaging app"),
    (r"\b(parcel|delivery|courier)\b.*\b(fee|pay|customs|held|reschedule|confirm)\b",
     0.15, "Delivery message asking for a fee or confirmation"),
]


def text_score(text: str) -> tuple[float, list[str]]:
    """
    Score an email or SMS for scam language.
    Returns (score between 0 and 1, list of reasons that fired).
    """
    lowered = text.lower()
    total = 0.0
    reasons = []

    for pattern, weight, reason in RULES:
        if re.search(pattern, lowered):
            total += weight
            reasons.append(reason)

    # Cap at 1.0 so many weak signals can't exceed the maximum
    return min(total, 1.0), reasons


if __name__ == "__main__":
    tests = [
        "Your order has shipped. Track it at the app.",
        "URGENT: your SBI account is locked. Verify your account details now.",
        "Congratulations! You won a lottery. Pay a processing fee to claim it.",
        "Pay the Rs 500 fine within 24 hours or face arrest: http://paypa1.com/pay",
    ]
    for t in tests:
        score, reasons = text_score(t)
        print(f"\n{t}\n -> {score:.2f}")
        for r in reasons:
            print("    -", r)