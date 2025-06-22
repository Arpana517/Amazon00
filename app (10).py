import pickle
import re
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
import os

# Download stopwords if not already downloaded
try:
    stopwords = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stopwords = set(stopwords.words('english'))


# Load the trained model, vectorizer, and scaler
try:
    with open('Models/best_rf_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('Models/countVectorizer.pkl', 'rb') as vectorizer_file:
        cv = pickle.load(vectorizer_file)
    with open('Models/scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    print("Please ensure 'best_rf_model.pkl', 'countVectorizer.pkl', and 'scaler.pkl' are in the 'Models' directory.")
    model, cv, scaler = None, None, None # Set to None if files are missing


# Initialize the stemmer
stemmer = PorterStemmer()

def preprocess_text(text):
    """
    Preprocesses the input text: removes non-alphabetic characters, converts to lowercase,
    stems words, and removes stopwords.
    """
    if isinstance(text, str):
        review = re.sub('[^a-zA-Z]', ' ', text)
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if word not in stopwords]
        return ' '.join(review)
    else:
        return '' # Return empty string for non-string input

def predict_sentiment(text):
    """
    Predicts the sentiment (positive or negative) of the input text using the loaded model.
    """
    if model is None or cv is None or scaler is None:
        return "Error: Model files not loaded."

    # Preprocess the input text
    preprocessed_text = preprocess_text(text)

    # Vectorize the preprocessed text
    text_vector = cv.transform([preprocessed_text]).toarray()

    # Scale the vectorized text
    text_vector_scaled = scaler.transform(text_vector)

    # Make prediction
    prediction = model.predict(text_vector_scaled)

    # Return sentiment label
    return "Positive" if prediction[0] == 1 else "Negative"

if __name__ == '__main__':
    # Example usage
    test_reviews = [
        "This is a great product! I love it.",
        "It stopped working after a week.",
        "Average quality, nothing special.",
        "Highly recommended!",
        "Very disappointed with the performance."
    ]

    print("Sentiment Predictions:")
    for review in test_reviews:
        sentiment = predict_sentiment(review)
        print(f"Review: '{review}' -> Sentiment: {sentiment}")
