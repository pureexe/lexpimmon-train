####
# validation_debug.py
# compare our model with baseline model
# Take time: haven't measure yet.
#

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from model_our import get_our_model
from model_baseline import get_model_baseline
from io_helper import pad_data, str2numpy
import pandas as pd
import argparse, os, shutil
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np

def main(args):
    testcsv_path = os.path.join(args.input,'validation.csv')
    if not os.path.exists(testcsv_path):
        raise RuntimeError("Cannot find CSV testfile. Have you run 'data_split.py' yet?")
    if not os.path.exists(args.model):
        raise RuntimeError("Cannot find model checkpoint. Have you run 'train.py' yet?")
    
    print("reading testfile csv... take about 5 seconds")
    test_df = pd.read_csv(testcsv_path)
    print("parsing testfile csv... take about 30 seconds")
    for column in test_df.columns[1:]:
        test_df[column] = test_df[column].apply(str2numpy)
    test_data = pad_data(test_df,'misskey_position')
    test_label = pad_data(test_df,'correct_label')
    
    print("Our Model")
    model_train, model_inference = get_our_model()
    model_train.load_weights(args.model)
    output = model_inference.predict(test_data)
    get_len = lambda x: np.count_nonzero(np.sum(x,axis = 1))
    predict_true = []
    predict_false = []
    for i in range(output.shape[0]):
        word_len = get_len(output[i])
        if ((output[i] == test_label[i]).all()):
            predict_true.append(word_len)
        else:
            predict_false.append(word_len)
    plt.hist(predict_true, bins = range(22))
    plt.title('Correct predict')
    plt.xlabel('Word length')
    plt.ylabel('Counts')
    plt.show()

    plt.hist(predict_false, bins = range(22))
    plt.title('Incorrect predict')
    plt.xlabel('Word length')
    plt.ylabel('Counts')
    plt.show()


def entry_point():
    parser = argparse.ArgumentParser(
        description='evaluation.py - measurement our network ')
    parser.add_argument(
        '-i',
        '--input',
        default='data/',
        type=str,
        help='path to csv testfile directory (default: data/)',
    )
    parser.add_argument(
        '--model',
        default='model/our_dense_activation_sigmoid/',
        type=str,
        help='path of model check point to load the weight (default: model/our/)',
    )
    parser.add_argument('--disable-baseline', 
        dest='disable_baseline', 
        action='store_true',
        help='use this option to measure only')
    parser.set_defaults(disable_baseline=True)
    main(parser.parse_args())

if __name__ == "__main__":
    entry_point()
