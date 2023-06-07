import json
import random
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
intents_file = "intents.json"
data = {"intents": []}
words = []
classes = []
documents = []
ignore_words = ["?", "¡", "!", ",", "."]

def load_intents():
    global data
    with open(intents_file) as file:
        data = json.load(file)

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in ignore_words]
    return tokens

def create_training_data():
    global words, classes
    for intent in data["intents"]:
        tag = intent["tag"]
        classes.append(tag)
        for pattern in intent["patterns"]:
            tokens = preprocess_text(pattern)
            words.extend(tokens)
            documents.append((tokens, tag))

    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    training_data = []
    output_data = []

    for document in documents:
        bag = []
        patterns = document[0]
        tag = document[1]
        for word in words:
            bag.append(1 if word in patterns else 0)

        output_row = [0] * len(classes)
        output_row[classes.index(tag)] = 1

        training_data.append(bag)
        output_data.append(output_row)

    return np.array(training_data), np.array(output_data)


def create_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(128, input_shape=(input_size,), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(output_size, activation="softmax"))

    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

    return model

def train_model():
    global words, classes
    training_data, output_data = create_training_data()
    model = create_model(len(words), len(classes))
    model.fit(training_data, output_data, epochs=200, batch_size=5, verbose=1)
    model.save("model.pkl")
    pickle.dump(words, open("words.pkl", "wb"))
    pickle.dump(classes, open("classes.pkl", "wb"))

load_intents()
train_model()
