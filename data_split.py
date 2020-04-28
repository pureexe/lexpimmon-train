####
# data_split.py
# split csv into train set, validation set and test set
# Take time: ~1 mins to run
#
import pandas as pd
import os,argparse
from timeit import default_timer as timer

def main(args):
    start_timer = timer()
    if not os.path.exists(args.input):
        raise RuntimeError('Cannot find csv input file. Have you run data_build.py before this file?')
    tdf = pd.read_csv(args.input)
    tdf.drop(tdf.columns[[0]], axis=1, inplace=True)
    # split into train by ratio, validation and test is the half from remain data
    train_df = tdf.sample(frac=args.ratio)
    test_valid_df = tdf.drop(train_df.index)
    validation_df = test_valid_df.sample(frac=0.5) #For example, half of 0.2 is 0.1
    test_df = test_valid_df.drop(validation_df.index)
    train_df.to_csv(os.path.join(args.output,'train.csv'))
    validation_df.to_csv(os.path.join(args.output,'validation.csv'))
    test_df.to_csv(os.path.join(args.output,'test.csv'))
    print('Finished in {:.2f} seconds'.format(timer() - start_timer))
    print('Output is write to {}'.format(os.path.abspath(args.output)))

def entry_point():
    parser = argparse.ArgumentParser(
        description='data_split.py - split csv into train.csv / vaidation.csv / test.csv')
    parser.add_argument(
        '-i',
        '--input',
        default='data/thaiword_training_data.csv',
        type=str,
        help='csv file of keyboard behavior (default: data/thaiword_training_data.csv)',
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='data/',
        help='output directory of train/validation/test csv file (default: data/ )')
    parser.add_argument(
        '-r',
        '--ratio',
        type=float,
        default=0.8,
        help='training set sample ratio, for validation and test will split the remain data from training test by the half (default: 0.8)')
    main(parser.parse_args())

if __name__ == "__main__":
    entry_point()