import streamlit as st
import pickle
import os
import pandas as pd
import numpy as np

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection System")
st.write(
    "This application predicts whether a news article is **likely Fake or Real** "
    "using a Machine Learning model trained on diverse news content."
)

st.markdown("---")

# ---------------- Load Model ----------------
def load_model():
    if not os.path.exists("model.pkl"):
        st.error("❌ model.pkl not found in the repository root")
        st.stop()
    if not os.path.exists("vectorizer.pkl"):
        st.error("❌ vectorizer.pkl not found in the repository root")
        st.stop()

    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# ---------------- User Input ----------------
news_text = st.text_area(
    "📝 Paste News Headline or Article",
    height=200,
    placeholder="Enter any news text (politics, health, technology, sports, etc.)"
)

# ---------------- Prediction ----------------
if st.button("🔍 Check News"):
    if news_text.strip() == "":
        st.warning("Please enter some news text.")
    else:
        text_vec = vectorizer.transform([news_text])

        prediction = model.predict(text_vec)[0]
        probabilities = model.predict_proba(text_vec)[0]
        confidence = probabilities.max()

        st.markdown("## 📊 Prediction Metrics")

        # ---- Metrics ----
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Predicted Class",
                "Fake News" if prediction == 0 else "Real News"
            )

        with col2:
            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

        with col3:
            st.metric(
                "Model",
                "Naive Bayes"
            )

        st.markdown("---")

        # ---- Final Decision ----
        if confidence < 0.65:
            st.warning(
                f"⚠️ **Uncertain – Needs Verification**  \n"
                f"Confidence: {confidence*100:.2f}%"
            )
        elif prediction == 0:
            st.error(
                f"🚨 **Likely Fake News**  \n"
                f"Confidence: {confidence*100:.2f}%"
            )
        else:
            st.success(
                f"✅ **Likely Real News**  \n"
                f"Confidence: {confidence*100:.2f}%"
            )

        # ---- Graph ----
        st.markdown("## 📈 Probability Distribution")

        prob_df = pd.DataFrame({
            "Class": ["Fake News", "Real News"],
            "Probability": probabilities
        })

        st.bar_chart(prob_df.set_index("Class"))

st.markdown("---")

st.caption(
    "⚠️ Disclaimer: This system provides probabilistic predictions based on linguistic patterns. "
    "It does not verify factual accuracy or news sources."
)
