####
# train.py
# train our model (that implement inside model_our.py)
# Take time: ~2 hours (1 card RTX 2080Ti)
#
# making test train, define metric ,what model to apply, turning

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from model_our import get_our_model
from io_helper import pad_data, str2numpy
import pandas as pd
import argparse, os, shutil

def get_callback(model_dir, log_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    return [
        TensorBoard(log_dir=log_dir, histogram_freq=10, write_graph=True),
        ModelCheckpoint(model_dir, save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min',verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001),
        EarlyStopping(monitor='val_loss', patience=20)    
    ]

def main(args):
    train_path = os.path.join(args.input,'train.csv')
    validation_path = os.path.join(args.input,'validation.csv')
    if not os.path.exists(train_path) or not os.path.exists(validation_path):
        raise RuntimeError("Cannot find train.csv. Have you run 'data_split.py' yet?")
    if args.restart:
        shutil.rmtree(args.model_dir,ignore_errors=True)
        shutil.rmtree(args.log_dir,ignore_errors=True)
        
    print("reading CSV... It's will take about 2 mins")
    train_df = pd.read_csv(train_path)    
    validation_df = pd.read_csv(validation_path)   

    print("parsing CSV... It's will take about 6 mins")
    for column in train_df.columns[1:]:
        train_df[column] = train_df[column].apply(str2numpy)
        validation_df[column] = validation_df[column].apply(str2numpy)
    train_data = pad_data(train_df,'misskey_position')
    train_label = pad_data(train_df,'correct_position')
    validation_data = pad_data(validation_df,'misskey_position')
    validation_label = pad_data(validation_df,'correct_position')

    print("start training")
    model_train, model_inference  = get_our_model(learning_rate = 0.01)
    if os.path.exists(args.model_dir):
        try:
            model_train.load_weights(args.model_dir)
        except:
            pass
    model_train.fit(
        train_data, 
        train_label, 
        epochs=1000,
        batch_size=int(train_data.shape[0] / 200),
        callbacks=get_callback(args.model_dir,args.log_dir), 
        validation_data=(validation_data, validation_label)
    )

def entry_point():
    parser = argparse.ArgumentParser(
        description='train.py - training file for our keyboard typing correction')
    parser.add_argument(
        '-i',
        '--input',
        default='data/',
        type=str,
        help='train & validation directory (default: data/)',
    )
    parser.add_argument(
        '--model_dir',
        default='model\\our\\',
        type=str,
        help='model directory (default: model\\our\\)',
    )
    parser.add_argument(
        '--log_dir',
        default='logs\\our\\',
        type=str,
        help='log directory (default: logs\\our\\)',
    )
    parser.add_argument('--restart', dest='restart', action='store_true')
    parser.set_defaults(restart=False)
    main(parser.parse_args())

if __name__ == "__main__":
    entry_point()