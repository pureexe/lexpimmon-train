import tensorflow as tf
from keras_helper import KeyboardPosition, RemoveTailNoise, word_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def get_model_baseline():
    input_node = Input(shape=(21,3))
    keyboard_position = KeyboardPosition()(input_node)
    clean_keyboard_layout = RemoveTailNoise()([keyboard_position, input_node])
    model = Model(inputs=input_node, outputs=clean_keyboard_layout)
    model.compile(loss='mean_squared_error', metrics=[word_accuracy])
    return model