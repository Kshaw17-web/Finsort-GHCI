# finsort/inference.py
import os
import json
import time
import joblib
import numpy as np

# local cleaner import
from .cleaner import clean_transaction, normalize_for_rules

# ---- Configurable paths ----
BASE = os.path.dirname(__file__)
_MODEL_PATH = os.path.join(BASE, "model.pkl")
_VECT_PATH = os.path.join(BASE, "vectorizer.pkl")
_CONFIG_PATH = os.path.join(BASE, "config.json")

# ---- Globals that get lazy-loaded and reloaded on change ----
_model = None
_vectorizer = None
_model_mtime = None
_vectorizer_mtime = None
_CONFIG = None

# ---------- Fast rule-based overrides (quick fix) -------------
MERCHANT_MAP = {
    # ecommerce / marketplaces
    "amazon": ("ecommerce", "Shopping"),
    "amzn": ("ecommerce", "Shopping"),
    "amazon marketplace": ("ecommerce", "Shopping"),
    "flipkart": ("ecommerce", "Shopping"),
    "myntra": ("ecommerce", "Shopping"),
    "ajio": ("ecommerce", "Shopping"),
    "meesho": ("ecommerce", "Shopping"),
    "nykaa": ("ecommerce", "Shopping"),
    # payments / gateways
    "paytm": ("ecommerce", "Shopping"),
    "phonepe": ("wallet", "Wallet"),
    "gpay": ("wallet", "Wallet"),
    "google pay": ("wallet", "Wallet"),
    "mobikwik": ("wallet", "Wallet"),
    # food / delivery
    "zomato": ("dining", "Dining"),
    "swiggy": ("dining", "Dining"),
    "dominos": ("dining", "Dining"),
    "pizzahut": ("dining", "Dining"),
    # coffee
    "starbucks": ("coffee_shop", "Food & Dining"),
    "coffee day": ("coffee_shop", "Food & Dining"),
    "ccd": ("coffee_shop", "Food & Dining"),
    "barista": ("coffee_shop", "Food & Dining"),
    # transport
    "ola": ("transport", "Transport"),
    "uber": ("transport", "Transport"),
    # fuel
    "hpcl": ("fuel", "Fuel"),
    "indianoil": ("fuel", "Fuel"),
    "bpcl": ("fuel", "Fuel"),
    # entertainment
    "netflix": ("entertainment", "Entertainment"),
    "prime video": ("entertainment", "Entertainment"),
    "spotify": ("entertainment", "Entertainment"),
    "bookmyshow": ("entertainment", "Entertainment"),
    # pharmacy
    "apollo": ("pharmacy", "Pharmacy"),
    "apollopharmacy": ("pharmacy", "Pharmacy"),
    "1mg": ("pharmacy", "Pharmacy"),
    # grocery
    "bigbasket": ("grocery", "Groceries"),
    "grofers": ("grocery", "Groceries"),
    "dmart": ("grocery", "Groceries"),
    # fallback
    "refund": ("refund", "Refund"),
    "reversal": ("refund", "Refund"),
}
GROCERY_KEYWORDS = {
    "tomato", "potato", "onion", "vegetable", "fruit",
    "milk", "bread", "eggs", "rice", "atta"
}

def rule_override(cleaned_text, config):
    """
    Quick rule-based classifier before ML model prediction.
    Returns dict with tag/category/confidence/low_confidence/by_rule or None.
    """
    txt = (cleaned_text or "").lower()
    cfg_map = (config or {}).get("category_map", {})

    # merchant contains match
    for k, (tag, cat) in MERCHANT_MAP.items():
        if k in txt:
            final_cat = cfg_map.get(tag, cat)
            return {
                "tag": tag,
                "category": final_cat,
                "confidence": 0.99,
                "low_confidence": False,
                "by_rule": True
            }

    # grocery keywords
    tokens = set(txt.split())
    if tokens & GROCERY_KEYWORDS:
        final_cat = cfg_map.get("grocery", "Groceries")
        return {
            "tag": "grocery",
            "category": final_cat,
            "confidence": 0.95,
            "low_confidence": False,
            "by_rule": True
        }

    # refund / credit patterns
    if "refund" in txt or txt.startswith("cr ") or "ref" in txt:
        final_cat = cfg_map.get("refund", "Refund")
        return {
            "tag": "refund",
            "category": final_cat,
            "confidence": 0.95,
            "low_confidence": False,
            "by_rule": True
        }

    return None

# --------------------------------------------------------------

def _load(force=False):
    """
    Load model, vectorizer and config if not loaded or if files changed on disk.
    Call at start of predict_category.
    """
    global _model, _vectorizer, _model_mtime, _vectorizer_mtime, _CONFIG

    # load config
    try:
        if _CONFIG is None or force:
            if os.path.exists(_CONFIG_PATH):
                with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                    _CONFIG = json.load(f)
            else:
                _CONFIG = {}
    except Exception:
        _CONFIG = {}

    # load model
    if os.path.exists(_MODEL_PATH):
        mtime = os.path.getmtime(_MODEL_PATH)
        if _model is None or _model_mtime != mtime or force:
            _model = joblib.load(_MODEL_PATH)
            _model_mtime = mtime

    # load vectorizer
    if os.path.exists(_VECT_PATH):
        vmtime = os.path.getmtime(_VECT_PATH)
        if _vectorizer is None or _vectorizer_mtime != vmtime or force:
            _vectorizer = joblib.load(_VECT_PATH)
            _vectorizer_mtime = vmtime

def _is_low_confidence(tag, confidence, config):
    default = config.get("confidence_threshold", 0.60)
    per = config.get("per_tag_threshold", {})
    tag_thresh = per.get(tag, default)
    return confidence < tag_thresh

def _ml_predict(cleaned_text):
    """
    Run ML model prediction (assumes _load() has ensured model & vectorizer exist).
    Returns dict with tag, category, confidence and low_confidence flag.
    """
    global _model, _vectorizer, _CONFIG
    if _model is None or _vectorizer is None:
        return None

    X = _vectorizer.transform([cleaned_text])
    probs = _model.predict_proba(X)[0]
    idx = int(np.argmax(probs))
    tag = _model.classes_[idx] if hasattr(_model, "classes_") else str(idx)
    confidence = float(probs[idx])
    # map tag -> category via config if available
    cfg_map = (_CONFIG or {}).get("category_map", {})
    category = cfg_map.get(tag, tag)
    low_conf = _is_low_confidence(tag, confidence, _CONFIG or {})
    return {"tag": tag, "category": category, "confidence": confidence, "low_confidence": low_conf}

def predict_category(raw_text):
    """
    Public inference function used by demo and scripts.
    Returns a dict: raw, cleaned, tag, category, confidence, low_confidence, maybe by_rule.
    """
    _load()  # ensure latest model/vectorizer/config

    raw = raw_text or ""
    cleaned = clean_transaction(raw)
    # normalized for rules (optional)
    cleaned_for_rules = normalize_for_rules(cleaned)

    # 1) Rule override
    r = rule_override(cleaned_for_rules, _CONFIG)
    if r:
        return {
            "raw": raw,
            "cleaned": cleaned,
            "tag": r["tag"],
            "category": r["category"],
            "confidence": float(r["confidence"]),
            "low_confidence": bool(r["low_confidence"]),
            "by_rule": True
        }

    # 2) ML fallback
    ml = _ml_predict(cleaned)
    if ml is None:
        # model not available; default safe return
        return {
            "raw": raw,
            "cleaned": cleaned,
            "tag": "unknown",
            "category": "Unknown",
            "confidence": 0.0,
            "low_confidence": True
        }

    result = {
        "raw": raw,
        "cleaned": cleaned,
        "tag": ml["tag"],
        "category": ml["category"],
        "confidence": ml["confidence"],
        "low_confidence": ml["low_confidence"]
    }
    return result

if __name__ == "__main__":
    tests = [
        "AMZN MKTP AY12B3 *PRIME",
        "STARBUCKS INDIA *STAR 09",
        "tomato 2kg",
        "REFUND AMAZON ORDER#77882",
        "PAYTM BILLPAY EB 093"
    ]
    for t in tests:
        try:
            print(t, "=>", predict_category(t))
        except Exception as e:
            print("Error for", t, e)
