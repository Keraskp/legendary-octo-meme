from numpy import array
from json import load
from nltk_utils import bag_of_words, tokenize, stem

import pickle
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import warnings
# warnings.filterwarnings('ignore')
# from tensorflow import keras
from keras import layers, Sequential

def prepare_dataset():
    with open('intents.json', 'r') as f:
        intents = load(f)

    all_words = []
    tags = []
    xy = []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))


    ignore_words = ['?', '.', '!']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    x_train = []
    y_train = []

    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        x_train.append(bag)

        label = tags.index(tag)
        y_train.append(label)

    x_train = array(x_train)
    y_train = array(y_train)

    return x_train, y_train, tags, all_words, len(tags)


def build_model(output_classes):
    model = Sequential([
        layers.Dense(16, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(output_classes, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam')
    return model


if __name__ == "__main__":
    x_train, y_train, tags, all_words,output_classes = prepare_dataset()
    model = build_model(output_classes)
    history = model.fit(x_train, y_train, epochs=100, batch_size=8, verbose=0)

    model.save('./model.h5')

    pickle.dump({'output_size': len(tags), 'input_size': len(x_train[0]),
                 'hidden_size': 32, 'all_words': all_words,
                 'tags': tags, 'x_train': x_train, 'y_train': y_train}, open("data.obj", "wb"))
