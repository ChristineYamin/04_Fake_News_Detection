import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
model = joblib.load("C:/Projects/04_Fake_News_Detection/models/svm_model.pkl")
vectorizer = joblib.load("C:/Projects/04_Fake_News_Detection/models/tfidf_vectorizer.pkl")

st.title("Fake News Detection App")

st.write("Enter a news article below to check whether it is fake or real")

user_input = st.text_area("Paste article text here")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform input
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]

        if hasattr(model, "predict_proba"):
             confidence = np.max(model.predict_proba(input_tfidf))
        else:
            confidence = "N/A"

        if prediction == 1:
            st.error(f"Prediction: Fake NEWS")
        else:
            st.success(f"Prediction: Real News")
        st.write(f"Confidence: {confidence}")