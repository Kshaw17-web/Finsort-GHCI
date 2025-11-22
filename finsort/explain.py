from .cleaner import clean_transaction
import joblib, os
BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "model.pkl")
VECT_PATH = os.path.join(BASE, "vectorizer.pkl")

def explain_prediction(text: str, top_k: int = 5):
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECT_PATH)
    except Exception:
        return []
    cleaned = clean_transaction(text)
    vec = vectorizer.transform([cleaned])
    if hasattr(model, 'coef_'):
        pred = model.predict(vec)[0]
        idx = list(model.classes_).index(pred)
        feature_names = vectorizer.get_feature_names_out()
        coefs = model.coef_[idx]
        nz = vec.nonzero()[1]
        word_weights = [(feature_names[i], float(coefs[i])) for i in nz]
        word_weights = sorted(word_weights, key=lambda x: x[1], reverse=True)
        return word_weights[:top_k]
    return []
