import os
import pandas as pd
from tqdm import tqdm

BASE = os.path.dirname(__file__)
TEST_CSV = os.path.join(BASE, "data", "finsort_test_hard20.csv")
OUT_DIR = os.path.join(BASE, "reports")
os.makedirs(OUT_DIR, exist_ok=True)

from finsort.inference import predict_category

df = pd.read_csv(TEST_CSV)

preds = []
for i, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
    tx = row.get("transaction", "")
    try:
        res = predict_category(tx)
    except Exception as e:
        res = {"tag": None, "category": None, "confidence": None, "error": str(e)}
    preds.append(res)

pred_df = pd.json_normalize(preds)
combined = pd.concat([df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)

combined["expected_tag"] = combined.get("expected_tag", None)
combined["match_tag"] = combined["tag"] == combined["expected_tag"]

results_csv = os.path.join(OUT_DIR, "hard_test_results.csv")
mismatch_csv = os.path.join(OUT_DIR, "hard_test_mismatches.csv")

combined.to_csv(results_csv, index=False)
combined[combined["match_tag"] != True].to_csv(mismatch_csv, index=False)

total = len(combined)
correct = combined["match_tag"].sum()
incorrect = total - correct

print(f"Total rows: {total}")
print(f"Correct predictions: {correct}")
print(f"Incorrect predictions: {incorrect}")
print("Full results saved to:", results_csv)
print("Mismatches saved to:", mismatch_csv)
