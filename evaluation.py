import pandas as pd, joblib, os
from sklearn.metrics import classification_report, confusion_matrix
from finsort.cleaner import clean_transaction
BASE = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE, 'finsort', 'model.pkl'))
vectorizer = joblib.load(os.path.join(BASE, 'finsort', 'vectorizer.pkl'))

df = pd.read_csv(os.path.join(BASE, 'data', 'finsort_test.csv'))
df['cleaned'] = df['transaction'].apply(clean_transaction)
X = vectorizer.transform(df['cleaned'])
preds = model.predict(X)
print(classification_report(df['tag'], preds))
print('Confusion matrix:')
print(confusion_matrix(df['tag'], preds))
