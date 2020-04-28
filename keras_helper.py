import tensorflow as tf
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K

def KeyboardPosition():
  """
  convert position from continous [-1,1] to discreate
  """
  def denormalize_coordinate_fn(x,shape = [2, 4, 12]):
    shape = tf.convert_to_tensor(shape,dtype=tf.float32)
    x = tf.math.round((x - (1.0 / shape) + 1.0) * (shape / 2.0))
    return x
  return Lambda(denormalize_coordinate_fn)

def RemoveTailNoise():
  """
  remove unuse info (noise) from vector. 
  For example, we use vector size  21 but our word lenght is just only 3
  we will set other 18 unused data to 0.0 from comparison
  """
  def zeros_mask(x):
    position, input_node = x
    mask = tf.not_equal(input_node,tf.zeros_like(position))
    mask = tf.cast(mask,tf.float32)
    position = tf.math.multiply(position,mask)
    return position
  return Lambda(zeros_mask)

def word_accuracy(y_true, y_pred):
    """
    Our custom metric for comparision between baseline and our model
    If all typing key are correct return 1 else 0
    Note: This is difference from Keras's accuracy metric. 
    If word has 3 length and 1 key is faild. Keras's accuracy metric will return 0.6667
    but we want it to return 0
    """
    # if word is same, the sum will return 0
    def is_correct_word(x):
        return K.sum(x)
    absolute = K.abs(y_true-y_pred)
    count = K.map_fn(is_correct_word,absolute)
    return K.equal(count,0)