import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

from wordlist_loader import WordListLoader

MODEL_PATH = 'model'

def main():
    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)
    word_dataset = WordListLoader()
    features = np.array(word_dataset.get_features(6))
    labels = np.array(word_dataset.get_labels(6))
    label_len = labels.shape[0]
    # remove layer and row from features
    features = features[:,:,2]
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=[6]),
        keras.layers.GaussianNoise(0.25), #it's always has noise in input
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(label_len)
    ])
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_PATH,'wordlen6.h5'),
            period=100
        )
    ]
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    model.fit(features, labels, epochs=10000)

    # test on training set lol (but this one don't has noise)
    test_loss, test_acc = model.evaluate(features,  labels, verbose=2)
    print('\nTest accuracy:', test_acc)
    #sample predict
    probability_model = keras.Sequential([model, keras.layers.Softmax()])
    predictions = probability_model.predict(features[0])
    predicted_label = np.argmax(predictions[0])
    print("Expected 0 predicted {}".format(predicted_label))


if __name__ == "__main__":
    main()