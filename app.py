mport streamlit as st
from transformers import pipeline

# Load the trained sentiment analysis model and tokenizer
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="sentiment_model",
    tokenizer="sentiment_model"
)

# Streamlit UI
st.title("Airline Review Sentiment Analyzer")
st.markdown("Enter an airline review below, and the app will predict its sentiment!")

# Input text box for user
user_input = st.text_area("Enter an airline review:")

# Button to analyze sentiment
if st.button("Analyze"):
    if user_input.strip():  # Ensure input is not empty
        result = sentiment_pipeline(user_input)
        sentiment = result[0]['label']
        confidence = result[0]['score']
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {confidence:.2f}")
    else:
        st.warning("Please enter some text to analyze.")
