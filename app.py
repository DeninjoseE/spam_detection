import streamlit as st
import pickle
import numpy as np
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the pre-trained model and TF-IDF vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf.pkl', 'rb') as tfidf_file:
    vectorizer = pickle.load(tfidf_file)

def predict_message(message):
    message_cleaned = preprocess_text(message)
    message_vectorized = vectorizer.transform([message_cleaned])
    prediction_proba = model.predict_proba(message_vectorized)
    prediction = model.predict(message_vectorized)
    return prediction_proba, 'Spam' if prediction[0] == 1 else 'Not spam'

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Add custom CSS
st.markdown("""
    <style>
    body {
        background-size: cover;
        background-position: center;
        color: #fff;
    }
    .stApp {
        background-image: url('https://cdn.analyticsvidhya.com/wp-content/uploads/2020/06/npl_spam-scaled.jpg');
        background-color: rgba(0, 0, 0, 0.7);
        # padding: 20px;
            
        # border-radius: 10px;
    }
    .stTextInput textarea {
        border-radius: 5px;
        padding: 10px;
        border: 1px solid #ddd;
        font-size: 16px;
        color: #fff;
        background-color: rgba(255, 255, 255, 0.1);
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stTitle, .stWrite, .stTextInput, .stButton, .stMarkdown {
        color: #00000;
            font-size: 16px;
        
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit application
st.title('SMS Spam Classification')

st.write('Enter your SMS message below to check if it is spam or not:')

user_input = st.text_area("Message")

if st.button('Classify'):
    if user_input:
        proba, result = predict_message(user_input)
        st.write(f'The message is classified as: **{result}**')
        st.write(f'Prediction probabilities: {proba}')
    else:
        st.write("Please enter a message.")
