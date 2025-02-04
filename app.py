import streamlit as st
from transformers import pipeline

# Load the model
pipe = pipeline("text-classification", model="iSathyam03/McD_Reviews_Sentiment_Analysis")

# Set up Streamlit app
st.title("McDonald's Review Sentiment Analysis")

# Create text input box for user to input review
review_text = st.text_area("Enter McDonald's Review:")

# Check if review text is not empty and run the sentiment analysis
if review_text:
    # Predict sentiment
    sentiment = pipe(review_text)
    
    # Extract the label and score
    sentiment_label = sentiment[0]['label']
    sentiment_score = sentiment[0]['score']
    
    # Display the result
    st.write(f"Sentiment: {sentiment_label}")
    st.write(f"Confidence: {sentiment_score:.2f}")

# Run the app
if __name__ == "__main__":
    st.write("Type a review in the box above to get sentiment analysis.")
