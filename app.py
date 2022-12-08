from flask import Flask, render_template, request, jsonify
from nltk_utils import bag_of_words, tokenize
import random
import json
import pickle
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import nltk
nltk.download('punkt')


app = Flask(__name__)

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

with open('data.obj', 'rb') as f:
    data = pickle.load(f)

all_words = data['all_words']
tags = data['tags']
model = tf.keras.models.load_model('./model.h5')


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def get_response(msg):
    global all_words, tags, model
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = np.array(X)

    output = model.predict(X, verbose=0)
    predicted = np.argmax(output)
    tag = tags[predicted]

    probs = softmax(output)
    prob = probs[0][predicted.item()]
    if prob.item() <= 0.75:
        return "I do not understand"
    for intent in intents['intents']:
        if tag == intent["tag"]:
            return random.choice(intent['responses'])


@app.get('/')
def index_get():
    return render_template("base.html")


@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    message = {'answer': response}
    return jsonify(message)


# if __name__ == "__main__":
#     app.run(debug=True)
