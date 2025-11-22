import joblib, json, os
from .cleaner import clean_transaction
BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "model.pkl")
VECT_PATH = os.path.join(BASE, "vectorizer.pkl")
CONFIG_PATH = os.path.join(BASE, "config.json")
# lazy load
_model = None
_vectorizer = None
_config = None

def _load():
    global _model, _vectorizer, _config
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _vectorizer is None:
        _vectorizer = joblib.load(VECT_PATH)
    if _config is None:
        try:
            with open(CONFIG_PATH, 'r') as f:
                _config = json.load(f)
        except Exception:
            _config = {}
    return _model, _vectorizer, _config

def predict_category(raw_text: str):
    model, vectorizer, config = _load()
    cleaned = clean_transaction(raw_text)
    vec = vectorizer.transform([cleaned])
    proba = model.predict_proba(vec)[0]
    tag = model.classes_[proba.argmax()]
    confidence = float(proba.max())
    final_category = config.get(tag, "Uncategorised")
    return {
        "raw": raw_text,
        "cleaned": cleaned,
        "tag": tag,
        "category": final_category,
        "confidence": confidence
    }
