# scripts/feedback_ingest.py
import os
import json
import pandas as pd

BASE = os.path.dirname(os.path.dirname(__file__))
LOG_PATH = os.path.join(BASE, "finsort", "feedback.log")
OUT_PATH = os.path.join(BASE, "data", "feedback.csv")

def load_feedback_entries():
    entries = []
    if not os.path.exists(LOG_PATH):
        print("No feedback.log at", LOG_PATH)
        return entries
    with open(LOG_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                entries.append(obj)
            except Exception:
                print("Skipping invalid JSON line:", line[:200])
    return entries

def normalize(entries):
    rows = []
    for e in entries:
        rows.append({
            "raw": e.get("raw"),
            "cleaned": e.get("cleaned"),
            "predicted_tag": e.get("predicted_tag") or e.get("tag"),
            "predicted_category": e.get("predicted_category") or e.get("category"),
            "confidence": e.get("confidence"),
            "corrected_category": e.get("corrected_category")
        })
    return pd.DataFrame(rows)

def main():
    print("Reading feedback log:", LOG_PATH)
    entries = load_feedback_entries()
    print("Loaded entries:", len(entries))
    if not entries:
        print("Nothing to write.")
        return
    df = normalize(entries)
    os.makedirs(os.path.join(BASE, "data"), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print("Wrote feedback CSV:", OUT_PATH)

if __name__ == "__main__":
    main()
