import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Load model and vectorizer
model = joblib.load("models/svm_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

st.title("Fake News Detection App")
st.write("Enter a news article below to check whether it is fake or real.")

# ---------- OPTIONAL: Proper testing using your dataset ----------
test_path = Path("clean_test.csv")

if "sample_text" not in st.session_state:
    st.session_state["sample_text"] = ""

col1, col2 = st.columns(2)

if test_path.exists():
    test_df = pd.read_csv(test_path)

    with col1:
        if st.button("Load random REAL example"):
            sample = test_df[test_df["label"] == 0].sample(1).iloc[0]
            st.session_state["sample_text"] = sample["combined_text"]

    with col2:
        if st.button("Load random FAKE example"):
            sample = test_df[test_df["label"] == 1].sample(1).iloc[0]
            st.session_state["sample_text"] = sample["combined_text"]
else:
    st.info("Tip: Put sample_test.csv next to app.py to enable random sample testing.")

# Text input
user_input = st.text_area("Paste article text here:", value=st.session_state["sample_text"], height=200)

def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_tfidf = vectorizer.transform([user_input])
        prediction = int(model.predict(input_tfidf)[0])

        # LinearSVC doesn't have predict_proba, so use decision_function
        score = float(model.decision_function(input_tfidf)[0])  # negative/positive distance
        confidence_like = sigmoid(score)  # 0..1 (NOT a true probability)

        if prediction == 1:
            st.error("Prediction: FAKE News")
        else:
            st.success("Prediction: REAL News")

        st.write(f"Decision score: {score:.3f}")
        st.write(f"Confidence-like (sigmoid): {confidence_like:.3f}")
        st.caption("Note: This is not a true probability. It indicates how far the input is from the decision boundary.")
