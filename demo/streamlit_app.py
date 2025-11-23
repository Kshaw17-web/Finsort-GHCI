# -*- coding: utf-8 -*-

import sys
import os
import json
import datetime
import streamlit as st

# Add project root so finsort package imports work
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from finsort.inference import predict_category

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "finsort", "config.json")
FEEDBACK_LOG = os.path.join(BASE_DIR, "finsort", "feedback.log")

st.set_page_config(page_title="FinSort Demo", layout="centered")

st.title("FinSort Demo")
st.write("Enter a transaction. The model will return a category and confidence score.")

tx = st.text_input("Transaction text", value="STARBUCKS INDIA *STAR 09")

if st.button("Predict"):

    try:
        res = predict_category(tx)
    except Exception as e:
        st.error("Prediction error: {}".format(e))
        st.stop()

    # Show the model outputs (clean display)
    st.subheader("Prediction Result")
    st.markdown("Predicted category: **{}**".format(res.get("category")))
    st.markdown("Confidence: **{:.2f}**".format(res.get("confidence")))

    if res.get("low_confidence"):
        st.warning("Low confidence. Please review and correct if needed.")

        # Load categories from config
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            all_cats = sorted(set(cfg.get("category_map", {}).values()))
        except Exception:
            all_cats = []

        choice = st.selectbox(
            "Correct category (if prediction is wrong):",
            ["(keep predicted)"] + all_cats
        )

        if st.button("Submit correction"):
            final_category = choice if choice != "(keep predicted)" else res.get("category")

            entry = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "raw": res.get("raw"),
                "predicted_category": res.get("category"),
                "corrected_category": final_category,
                "confidence": float(res.get("confidence"))
            }

            os.makedirs(os.path.dirname(FEEDBACK_LOG), exist_ok=True)
            with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

            st.success("Feedback recorded.")
    else:
        st.success("High confidence. No correction needed.")




