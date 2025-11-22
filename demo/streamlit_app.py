import streamlit as st
import os
from finsort.inference import predict_category
from finsort.explain import explain_prediction

st.set_page_config(page_title="FinSort - Transaction Categorization", page_icon="💰")

st.title("💰 FinSort - Transaction Categorization")

st.markdown("Enter a transaction string to predict its category.")

# Transaction input
transaction_input = st.text_input("Transaction string", placeholder="e.g., SQ *COFFEE-SPOT 123 or AMAZON MKTPLACE PMTS")

# Predict button
if st.button("Predict"):
    if transaction_input.strip():
        try:
            # Get prediction
            result = predict_category(transaction_input)
            
            # Display results
            st.subheader("Prediction Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Tag", result['tag'])
                st.metric("Category", result['category'])
            with col2:
                st.metric("Confidence", f"{result['confidence']:.2%}")
            
            # Show cleaned text
            st.info(f"Cleaned text: {result['cleaned']}")
            
            # Get top 4 explanation tokens
            explanations = explain_prediction(transaction_input, top_k=4)
            if explanations:
                st.subheader("Top 4 Explanation Tokens")
                exp_text = ", ".join([f"{word} ({weight:.3f})" for word, weight in explanations])
                st.code(exp_text)
            else:
                st.warning("Could not generate explanations (model may not be loaded).")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Make sure you have trained the model first by running: python train.py")
    else:
        st.warning("Please enter a transaction string.")

st.divider()

# Correction submission section
st.subheader("Submit Correction")
st.markdown("If the prediction was incorrect, please provide the correct category below.")

correction_input = st.text_input("Correct category", placeholder="e.g., Groceries, Food & Dining")

if st.button("Submit Correction"):
    if transaction_input.strip() and correction_input.strip():
        try:
            feedback_path = os.path.join(os.path.dirname(__file__), '..', 'finsort', 'feedback.log')
            with open(feedback_path, 'a') as f:
                f.write(f"{transaction_input},{correction_input}\n")
            st.success(f"Correction logged: '{transaction_input}' → '{correction_input}'")
        except Exception as e:
            st.error(f"Error logging correction: {str(e)}")
    elif not transaction_input.strip():
        st.warning("Please enter a transaction string first.")
    elif not correction_input.strip():
        st.warning("Please enter the correct category.")

st.divider()
st.caption("Note: Make sure model.pkl and vectorizer.pkl exist in finsort/ directory (run python train.py first).")

