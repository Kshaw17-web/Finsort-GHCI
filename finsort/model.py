import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from .cleaner import clean_transaction
import os

BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "model.pkl")
VECT_PATH = os.path.join(BASE, "vectorizer.pkl")

def train_model(csv_path: str = None, save=True):
    if csv_path is None:
        csv_path = os.path.join(BASE, "..", "data", "finsort_train.csv")
    df = pd.read_csv(csv_path)
    df['cleaned'] = df['transaction'].apply(clean_transaction)
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned'], df['tag'], test_size=0.2, random_state=42, stratify=df['tag']
    )
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = LogisticRegression(max_iter=500)
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    print("Classification report:\n", classification_report(y_test, preds))
    print("Macro F1:", f1_score(y_test, preds, average='macro'))
    if save:
        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECT_PATH)
        print(f"Saved model to {MODEL_PATH} and vectorizer to {VECT_PATH}")
    return model, vectorizer

if __name__ == '__main__':
    train_model()
