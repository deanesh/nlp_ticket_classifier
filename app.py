
import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences

stop_words = set(stopwords.words('english'))

# Load model and preprocessors
model = tf.keras.models.load_model('model/lstm_model.keras')

with open('model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('model/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Clean text
def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    words = text.split()
    return ' '.join([word for word in words if word not in stop_words])

# Streamlit UI
st.title("🛠️ Support Ticket Classifier")
st.write("Enter a customer support ticket message to classify it.")

st.write("App loaded ✅")
user_input = st.text_area("Ticket Text:")

if st.button("Classify"):
    try:
        cleaned = clean_text(user_input)
        st.write(cleaned)

        if not cleaned.strip():
            st.error("Input is empty or only contains stopwords. Please enter valid text.")
        else:
            seq = tokenizer.texts_to_sequences([cleaned])
            
            if not seq or not seq[0]:
                st.error("Input could not be processed. Try different or longer text.")
            else:
                padded = pad_sequences(seq, maxlen=20)
                pred = model.predict(padded)
                st.write(pred)
                label = label_encoder.inverse_transform([np.argmax(pred)])
                st.write(label)
                st.success(f"Predicted Category: **{label[0]}**")
    except Exception as ex:
        st.error("Input could not be processed. Try different or longer text.")
