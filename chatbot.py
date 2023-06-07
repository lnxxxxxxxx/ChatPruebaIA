import json
import random
import pickle
import numpy as np
from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
intents_file = "intents.json"
data = {"intents": []}
words = []
classes = []

def load_intents():
    global data
    with open(intents_file) as file:
        data = json.load(file)

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def load_model_files():
    global words, classes
    words = pickle.load(open("words.pkl", "rb"))
    classes = pickle.load(open("classes.pkl", "rb"))

def preprocess_input_text(text):
    tokens = preprocess_text(text)
    bag = [0] * len(words)
    for w in tokens:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def get_response(model, input_text):
    input_data = preprocess_input_text(input_text)
    results = model.predict(np.array([input_data]))[0]
    results_indices = np.argmax(results)
    tag = classes[results_indices]

    threshold = 0.9
    if results[results_indices] > threshold:
        tag = classes[results_indices]
        for intent in data["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
    else:
        for intent in data["intents"]:
            if intent["tag"] == "desconocido":
                return random.choice(intent["responses"])


load_intents()
load_model_files()
model = load_model("model.pkl")


# descomenta o comenta depende si quieres
# que funcione como un while como aplicacion
# o si quieres levantarla como api
# while True:
#     user_input = input("User: ")
#     if user_input.lower() == "quit":
#         break
    
#     response = get_response(model, user_input)
#     print("ChatBot:", response)
