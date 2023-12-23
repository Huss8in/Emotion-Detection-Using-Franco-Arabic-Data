from flask import Flask, render_template, request
from googletrans import Translator
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import re
import torch
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModel
import nltk
from keras.models import load_model
import joblib

# Load your LSTM model
lstm_model = load_model('A) Arabic approch/emotion_detection_model.h5')
label_encoder = joblib.load('A) Arabic approch/label_encoder.joblib')

app = Flask(__name__)

# Load AraBERT tokenizer and model
model_name = "aubmindlab/bert-base-arabertv02"
AraBERT_tokenizer = AutoTokenizer.from_pretrained(model_name)
AraBERT_model = AutoModel.from_pretrained(model_name)

# List of Arabic stopwords
arabic_stopwords = stopwords.words('arabic')


def clean_text(text):
    # Remove any non Arabic unicode
    text = re.sub(r'[^\u0600-\u06FF\s]+', ' ', text)
    # Remove username "@handle" from text
    text = re.sub(r'@\w+', '', text)
    # Remove URL from text
    text = re.sub(r'http\S+', '', text)
    # Remove punctuation, emoji, and smileys from text
    text = re.sub(r'(?<=\w)[^\s\w](?![^\s\w])', '', text)
    # Remove escape codes like \n, \t, \, etc from text
    text = re.sub(r'(\n|\t|\"|\')', '', text)
    # Remove Arabic Diacritization (tashkeel) like fatha, damma, kasra, shaddah, ...
    text = re.sub(r'[\u064B-\u0652]', '', text)
    # Removing Digits
    text = re.sub(r'\d', '', text)

    # Tokenize the text
    words = word_tokenize(text)
    # Remove Arabic stopwords
    words = [word for word in words if word not in arabic_stopwords]

    return ' '.join(words)

def get_arabert_embedding(text):
    tokens = AraBERT_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        model_output = AraBERT_model(**tokens)
    # Return the embeddings for the [CLS] token
    return model_output.last_hidden_state[:, 0, :].numpy()

def clean_text_Franco(text):
    # Remove URLs https://www.example.com/
    text = re.sub(r'http\S+', '', text)
    # Remove user mentions @mohamed_samy
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    # Remove hashtags #free_palestine
    text = re.sub(r'#', '', text)
    # Remove punctuation period, comma, apostrophe, quotation, question, exclamation, brackets, braces, parenthesis, dash, hyphen, ellipsis, colon, semicolon
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Convert to lower case
    tokens = [word.lower() for word in tokens]
    # print(tokens)
    # Join the tokens back into a clean text
    clean_text = ' '.join(tokens)

    return clean_text

def is_arabic(text):
    arabic_letters = set("ابتثجحخدذرزسشصضطظعغفقكلمنهويءآأؤإئى")
    return any(char in arabic_letters for char in text)

def translate_to_arabic(text):
    translator = Translator()
    translation = translator.translate(text, dest='ar')
    return translation.text

def test_model_with_probabilities(model, label_encoder, text):
    if not is_arabic(text):
        print("Input is Franco-Arabic!")
        print("\n")

        print(f"Original User Input: {text}")
        print("\n")

        clean_Franco = clean_text_Franco(text)
        print(f"Cleaned User Input: {clean_Franco}")
        print("\n")

        translated_user_input = translate_to_arabic(clean_Franco)
        print(f"Translated User Input: {translated_user_input}")
        print("\n")

        text = translated_user_input
    else:
        print("Input is Arabic!")
        print("\n")

        print(f"Original User Input: {text}")
        print("\n")

        clean_user_input = clean_text(text)
        print(f"Cleaned User Input: {clean_user_input}")
        print("\n")
        
        text = clean_user_input

    print("----------------------------------------------------------------")

    embedding = get_arabert_embedding(text)
    # Pad the sequence
    padded_sequence = pad_sequences([embedding], padding='post', dtype='float32')
    # Make predictions
    prediction_prob = model.predict(padded_sequence)
    predicted_class = np.argmax(prediction_prob)
    # Convert predicted class back to the original label
    predicted_label = label_encoder.classes_[predicted_class]
    # Get class probabilities
    class_probabilities = {label: prob for label, prob in zip(label_encoder.classes_, prediction_prob[0])}

    return predicted_label, class_probabilities

def analyze_user_input(user_input_text):
    predicted_emotion, class_probabilities = test_model_with_probabilities(lstm_model, label_encoder, user_input_text)

    # Calculate total probability
    total_probability = sum(class_probabilities.values())

    # Print predicted emotion and percentage of each class in descending order
    print(f"Predicted Emotion: {predicted_emotion}")
    print("----------------------------------------------------------------")
    print("Class Probabilities:")
    sorted_probabilities = dict(sorted(class_probabilities.items(), key=lambda item: item[1], reverse=True))
    for emotion, probability in sorted_probabilities.items():
        percentage = (probability / total_probability) * 100
        print(f"{emotion}: {percentage:.2f}%")
    print("----------------------------------------------------------------")

    # Plotting the results
    fig, ax = plt.subplots()
    ax.bar(sorted_probabilities.keys(), sorted_probabilities.values())
    ax.set_ylabel('Probability')
    ax.set_title(f'Emotion Prediction (Descending Order)')
    ax.tick_params(axis='x', rotation=45)

    # Save the plot to a BytesIO object
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    # Encode the image to base64 for HTML display
    image_base64 = base64.b64encode(image_stream.read()).decode('utf-8')

    return predicted_emotion, image_base64

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    predicted_emotion, image_base64 = analyze_user_input(user_input)

    return render_template('result.html', predicted_emotion=predicted_emotion, image_base64=image_base64)

if __name__ == '__main__':
    app.run(debug=True)
