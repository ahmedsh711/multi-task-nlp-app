import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import gdown

# --------------- MODEL DOWNLOAD & LOADING ---------------

MODEL_FILE = "MultiTaskLearning_NLP.keras"
DRIVE_FILE_ID = "12ZRglomEvtg73n5OeuldXqQ7d_Be5_bM"
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_FILE):
        st.write("Downloading model from Drive...")
        gdown.download(DRIVE_URL, MODEL_FILE, quiet=False)
    model = tf.keras.models.load_model(MODEL_FILE)
    return model

@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

# Load model & tokenizer
model = download_and_load_model()
tokenizer = load_tokenizer()

max_length = 50

# --------------- SIMPLE STOPWORD REMOVAL (NO NLTK) ---------------

custom_stopwords = {
    'a','an','the','and','or','but','if','while','is','am','are','was','were',
    'be','to','of','in','that','this','with','for','on','as','at','it','from',
    'by','about','so','just','into','like'
}

def remove_stopwords(text):
    words = text.split()
    filtered = [w for w in words if w.lower() not in custom_stopwords]
    return ' '.join(filtered)

# --------------- PREDICTION FUNCTION ---------------

def classify_text(input_text):
    input_text_cleaned = remove_stopwords(input_text)
    input_sequence = tokenizer.texts_to_sequences([input_text_cleaned])
    input_padded = pad_sequences(input_sequence, maxlen=max_length, padding='post')

    prediction = model.predict({
        'emotion_input': input_padded,
        'violence_input': input_padded,
        'hate_input': input_padded
    })

    # Task selection
    emotion_pred = np.argmax(prediction[0], axis=1)[0]
    violence_pred = np.argmax(prediction[1], axis=1)[0]
    hate_pred = np.argmax(prediction[2], axis=1)[0]

    major_labels = ['Emotion', 'Violence', 'Hate']
    major_label_index = np.argmax([
        np.max(prediction[0]),
        np.max(prediction[1]),
        np.max(prediction[2])
    ])
    major_label = major_labels[major_label_index]

    emotion_labels_text = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']
    violence_labels_text = [
        'Sexual_Violence',
        'Physical_Violence',
        'Emotional_Violence',
        'Harmful_Traditional_Practice',
        'Economic_Violence'
    ]
    hate_labels_text = ['Offensive Speech', 'Neither', 'Hate Speech']

    if major_label == 'Emotion':
        sub_label = emotion_labels_text[emotion_pred]
    elif major_label == 'Violence':
        sub_label = violence_labels_text[violence_pred]
    else:
        sub_label = hate_labels_text[hate_pred]

    # Confidence
    if major_label == 'Emotion':
        confidence = float(np.max(prediction[0]))
    elif major_label == 'Violence':
        confidence = float(np.max(prediction[1]))
    else:
        confidence = float(np.max(prediction[2]))

    return major_label, sub_label, confidence

# --------------- STREAMLIT UI ---------------

st.title("Multi-Task NLP Classifier (Emotion / Hate / Violence)")
st.write("Enter any text, and the model will classify it into Emotion, Hate Speech, or Violence.")

user_input = st.text_area("Enter your text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        major, sub, conf = classify_text(user_input)
        st.success(f"**Major Category:** {major}")
        st.info(f"**Sub Label:** {sub}")
        st.write(f"**Confidence:** {conf:.4f}")
