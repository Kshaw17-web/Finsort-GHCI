# finsort/cleaner.py
# Robust transaction text cleaning and normalization.

import re
import unicodedata

def clean_transaction(text):
    """
    Normalize transaction strings for model ingestion.

    Steps:
    - ensure str, lower-case
    - unicode normalize and remove control characters
    - replace common abbreviations and merchant aliases
    - remove long numeric sequences (IDs), keep short numbers (like '2kg')
    - remove common noise tokens
    - drop punctuation, collapse whitespace
    """
    if not text:
        return ""

    s = str(text)

    # 1. Unicode normalize & remove control characters
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")

    # 2. Lowercase
    s = s.lower()

    # 3. Common alias normalizations (regex keys)
    replacements = {
        r"\bamzn\b": "amazon",
        r"\bamz\b": "amazon",
        r"\bamzn mk?tp\b": "amazon marketplace",
        r"\bamzn mktp\b": "amazon marketplace",
        r"\bmktp\b": "marketplace",
        r"\bmkt\b": "marketplace",
        r"\bflipkart\b": "flipkart",
        r"\bmyntra\b": "myntra",
        r"\bajio\b": "ajio",
        r"\bpaytm billpay\b": "paytm billpay",
        r"\bpaytm\b": "paytm",
        r"\bphonepe\b": "phonepe",
        r"\bgpay\b": "google pay",
        r"\bgpay\b": "google pay",
        r"\bccd\b": "coffee day",
        r"\bstarbucks\b": "starbucks",
        r"\bnetflixcom\b": "netflix",
        r"\bnetflix\b": "netflix",
        r"\bspotify\b": "spotify",
        r"\bamazonpay\b": "amazon pay",
        r"\bupi[-/]?\b": "upi ",
        r"\bpayu\b": "payu",
    }
    for pat, repl in replacements.items():
        s = re.sub(pat, repl, s)

    # 4. Remove/normalize common noise tokens (but keep useful short numbers)
    noise_tokens = [
        r"\btxn\b", r"\btrx\b", r"\bref\b", r"\breference\b",
        r"\binv\b", r"\binvoice\b", r"\bord\b", r"\bpmts\b", r"\bpmts?\b",
        r"\bcr\b", r"\bdr\b", r"\bautopay\b", r"\bautopay\b", r"\btransfer\b",
        r"\bpaid\b", r"\bpayment\b"
    ]
    for nt in noise_tokens:
        s = re.sub(nt, " ", s)

    # 5. Remove long runs of digits (IDs, card fragments, phone numbers)
    s = re.sub(r"\d{6,}", " ", s)

    # 6. Remove stray special characters
    s = re.sub(r"[\*\#\@\!\$\%\^\&\(\)_\+\=\[\]\{\};:<>\/\\\|~`]", " ", s)

    # 7. Normalize separators / punctuation
    s = re.sub(r"[-_/,.]+", " ", s)

    # 8. Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s

def normalize_for_rules(cleaned):
    """
    Further lightweight normalization specifically for rule matching.
    """
    if not cleaned:
        return ""
    s = cleaned
    s = re.sub(r"^(to|for)\s+", "", s)
    s = re.sub(r"\s+(inc|ltd|pvt|india|co|company)$", "", s)
    s = s.strip()
    return s

if __name__ == "__main__":
    tests = [
        "AMZN MKTP AY12B3 *PRIME",
        "SQ *COFFEE-SPOT 123",
        "UPI-AXIS/9845123456-PAY",
        "tomato 2kg",
        "REFUND AMAZON ORDER#77882",
        "PAYTM BILLPAY EB 093"
    ]
    for t in tests:
        print("orig:", t)
        c = clean_transaction(t)
        print("clean:", c)
        print("---")
