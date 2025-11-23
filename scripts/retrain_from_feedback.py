# scripts/retrain_from_feedback.py
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(__file__))
MERGE = os.path.join(BASE, "scripts", "merge_feedback_to_train.py")
AUG = os.path.join(BASE, "data", "finsort_train_augmented.csv")
TRAIN_PY = os.path.join(BASE, "train.py")

def main():
    print("Merging feedback into augmented train...")
    subprocess.check_call([sys.executable, MERGE])

    if not os.path.exists(AUG):
        raise SystemExit("Augmented train file not found: " + AUG)

    env = os.environ.copy()
    env['TRAIN_PATH'] = AUG
    print("Training using augmented dataset:", AUG)
    subprocess.check_call([sys.executable, TRAIN_PY], env=env)
    print("Retrain finished. New model saved to finsort/model.pkl")

if __name__ == "__main__":
    main()
