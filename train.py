import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from wordlist_loader import WordListLoader
from keyboard_layout import KeyboardLayout

MODEL_PATH = 'model'

def main():
    for word_len in range(2,13):
        train_wordlen(word_len)

def get_model(word_len,label_len):
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=[word_len]),
        keras.layers.GaussianNoise(0.1), #it's always has noise in input
        keras.layers.Dense(128, activation='relu'),
#        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(label_len)
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    return model

def measure_testset():
    word_dataset = WordListLoader()
    for word_len in range(2,13):
        K.clear_session()
        features = np.array(word_dataset.get_features(word_len))
        labels = np.array(word_dataset.get_labels(word_len))
        label_len = labels.shape[0]
        features = features[:,:,2]
        check_point_path = os.path.join(MODEL_PATH,'wordlen_{:02d}.h5'.format(word_len))
        model = get_model(word_len,label_len)
        model.load_weights(check_point_path)
        test_loss, test_acc = model.evaluate(features,  labels, verbose=0)
        print('WordLen {:2d} Test accuracy: {:6f}'.format(word_len,test_acc))

def train_wordlen(word_len):
    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)
    K.clear_session()
    word_dataset = WordListLoader()
    features = np.array(word_dataset.get_features(word_len))
    labels = np.array(word_dataset.get_labels(word_len))
    label_len = labels.shape[0]
    # remove layer and row from features
    features = features[:,:,2]
    #features = features[:,:,1:3]
 
    check_point_path = os.path.join(MODEL_PATH,'wordlen_{:02d}.h5'.format(word_len))
    callbacks_list = [
        keras.callbacks.TensorBoard(log_dir='logs\\wordlen_{:02d}'.format(word_len), histogram_freq=1, write_graph=True, write_grads=True),
        keras.callbacks.ModelCheckpoint(
            check_point_path,
            save_weights_only=True,
            save_best_only=True,
            monitor='val_accuracy',
            mode='min',
        ),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001),
        keras.callbacks.EarlyStopping(monitor='accuracy',patience = 100)
    ]
    
    model = get_model(word_len,label_len)

    if os.path.exists(check_point_path):
        model.load_weights(check_point_path)
    
    model.fit(features, labels, callbacks=callbacks_list, epochs=100000, verbose=1 ,validation_data=(features,labels) , batch_size=label_len)

    # test on training set lol (but this one don't has noise)
    test_loss, test_acc = model.evaluate(features,  labels, verbose=2)
    print('\nTest accuracy:', test_acc)
    #sample predict
    probability_model = keras.Sequential([model, keras.layers.Softmax()])
    predictions = probability_model.predict(features[:1])
    predicted_label = np.argmax(predictions[0])

def test_do_you_eat():
    words = ['กิน','ข้าว','มา','แล้ว','หรือ','ยัง']
    word_dataset = WordListLoader()
    kb = KeyboardLayout()
    for word in words:
        K.clear_session()
        w_len = len(word)
        labels = np.array(word_dataset.get_labels(w_len))
        label_len = labels.shape[0]
        model = get_model(w_len,label_len)
        probability_model = keras.Sequential([model, keras.layers.Softmax()])
        word_np = np.array(kb.to_position(word,True))
        word_np = word_np[:,2]
        predictions = probability_model.predict(word_np.reshape(-1,w_len))
        prediction = np.argsort(-predictions[0])
        print("========")
        print(word)
        print(kb.to_position(word))
        print("top 5 prediction")
        print('---------')
        for i in range(5):
            predict_word = word_dataset.get_class_label(w_len,prediction[i])
            print(predict_word)
            print(kb.to_position(predict_word))

if __name__ == "__main__":
    #main()
    #measure_testset()
    test_do_you_eat()