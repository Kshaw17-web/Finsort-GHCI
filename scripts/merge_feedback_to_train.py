# scripts/merge_feedback_to_train.py
import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(__file__))
TRAIN = os.path.join(BASE, "data", "finsort_train.csv")
FEEDBACK = os.path.join(BASE, "data", "feedback.csv")
OUT = os.path.join(BASE, "data", "finsort_train_augmented.csv")

def feedback_to_train_rows(df_fb):
    out = []
    for _, r in df_fb.iterrows():
        transaction = r.get("raw") or r.get("transaction")
        corrected = r.get("corrected_category") or r.get("predicted_category") or r.get("predicted_tag")
        cleaned = r.get("cleaned", "")
        if not transaction or not corrected:
            continue
        out.append({"transaction": transaction, "cleaned": cleaned, "category": corrected})
    return pd.DataFrame(out)

def main():
    if not os.path.exists(TRAIN):
        raise SystemExit("Base train file not found at: " + TRAIN)
    train_df = pd.read_csv(TRAIN)

    if not os.path.exists(FEEDBACK):
        print("No feedback.csv found; copying original train to augmented.")
        train_df.to_csv(OUT, index=False)
        print("Wrote", OUT)
        return

    fb_df = pd.read_csv(FEEDBACK)
    fb_rows = feedback_to_train_rows(fb_df)
    if fb_rows.empty:
        print("No usable feedback rows.")
        train_df.to_csv(OUT, index=False)
        return

    combined = pd.concat([train_df, fb_rows], ignore_index=True)
    combined = combined.drop_duplicates(subset=["transaction","category"], keep="last")
    combined.to_csv(OUT, index=False)
    print("Wrote augmented train:", OUT, "rows:", len(combined))

if __name__ == "__main__":
    main()
