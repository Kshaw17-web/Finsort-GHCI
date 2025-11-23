# train.py
"""
Train script for FinSort.
- Reads training CSV from TRAIN_PATH (env var or data/finsort_train.csv)
- Expects CSV with 'transaction' and 'category' columns (or adapt)
- Trains a TF-IDF vectorizer and a calibrated logistic regression classifier
- Saves vectorizer.pkl and model.pkl into finsort/
"""

import os
import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE = os.path.dirname(__file__)
TRAIN_PATH = os.environ.get("TRAIN_PATH", os.path.join(BASE, "data", "finsort_train.csv"))
VECT_OUT = os.path.join(BASE, "finsort", "vectorizer.pkl")
MODEL_OUT = os.path.join(BASE, "finsort", "model.pkl")

def load_data(path):
    df = pd.read_csv(path)
    # Try to find right column names
    if "transaction" in df.columns:
        X = df["transaction"].astype(str).tolist()
    elif "raw" in df.columns:
        X = df["raw"].astype(str).tolist()
    else:
        # fallback: first column
        X = df.iloc[:,0].astype(str).tolist()

    if "category" in df.columns:
        y = df["category"].astype(str).tolist()
    elif "tag" in df.columns:
        y = df["tag"].astype(str).tolist()
    else:
        # fallback: last column
        y = df.iloc[:,-1].astype(str).tolist()
    return X, y

def main():
    print("Loading training data from:", TRAIN_PATH)
    X_texts, y = load_data(TRAIN_PATH)
    print("Rows:", len(X_texts))

    # preprocessing: use cleaned texts via finsort.cleaner if available
    try:
        from finsort.cleaner import clean_transaction
        X_texts = [clean_transaction(t) for t in X_texts]
    except Exception as e:
        print("Could not import clean_transaction:", e)

    # vectorizer (word + char ngrams works well for noisy text)
    vect = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=30000)
    X = vect.fit_transform(X_texts)
    print("Vectorized shape:", X.shape)

    # label encode
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # base classifier
    base = LogisticRegression(max_iter=1000, class_weight='balanced')
    # calibrate probabilities
    clf = CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=5)
    print("Training classifier (this may take a few minutes)...")
    clf.fit(X, y_enc)

    # Create a small wrapper to restore original labels on predict
    class WrappedModel:
        def __init__(self, model, label_encoder):
            self.model = model
            self.le = label_encoder
            # sklearn CalibratedClassifierCV keeps classes_ as encoded integers
            self.classes_ = self.le.inverse_transform(np.arange(len(self.le.classes_)))

        def predict_proba(self, X):
            return self.model.predict_proba(X)

        def predict(self, X):
            preds = self.model.predict(X)
            return self.le.inverse_transform(preds)

    wrapped = WrappedModel(clf, le)

    # Save vectorizer and model
    os.makedirs(os.path.join(BASE, "finsort"), exist_ok=True)
    joblib.dump(vect, VECT_OUT)
    joblib.dump(wrapped, MODEL_OUT)
    print("Saved vectorizer ->", VECT_OUT)
    print("Saved model ->", MODEL_OUT)

if __name__ == "__main__":
    main()
