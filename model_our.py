import tensorflow as tf
from keras_helper import KeyboardPosition, RemoveTailNoise, word_accuracy
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

def get_our_model(learning_rate = 0.001):
  input_node = Input(shape=(21,3))
  shift_controller = input_node[:,:,:1]
  x = Dense(256,activation='relu')(input_node)
  x = Dense(256,activation='relu')(x)
  x = Dense(256,activation='relu')(x)
  x = Dense(256,activation='relu')(x)
  x = Concatenate()([x,input_node])
  x = Dense(256,activation='relu')(x)
  x = Dense(256,activation='relu')(x)
  x = Dense(256,activation='relu')(x)
  x = Dense(2)(x)
  concat = Concatenate(axis=2)([shift_controller,x])
  keyboard_position = KeyboardPosition()(concat)
  cleand_keyboard_positioin = RemoveTailNoise()([keyboard_position,input_node])
  model_train = Model(inputs=input_node, outputs=concat)
  model_inference  = Model(inputs=input_node, outputs=cleand_keyboard_positioin)
  optimizer = Adam(learning_rate=learning_rate)
  model_train.compile(loss='mean_squared_error', optimizer=optimizer)
  model_inference.compile(loss='mean_squared_error', metrics=[word_accuracy])
  return model_train, model_inference 