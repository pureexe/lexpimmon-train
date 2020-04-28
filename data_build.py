####
# data_build.py
# build csv data of keyboard typing behavior
# Take time: ~30 mins to run
#

import random, os, argparse
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from keyboard_layout import KeyboardLayout

WORDLIST_FILE = 'data/'
CSV_OUTPUT = 'thai-wordlist.txt'

def create_dataset(wordlist, data_per_word = 100):
  datasets = []
  kbl = KeyboardLayout()
  kb_shape = kbl.shape()
  for word in wordlist:
    position = kbl.to_position(word)
    position_normalized = kbl.normalize(position)
    word_len = len(word)
    half_len = int(word_len/2)
    for i in range(data_per_word):
      misskey_position = position.copy()
      number_miss_type = random.randint(0,half_len)
      key_to_misstype = list(range(word_len))
      random.shuffle(key_to_misstype)
      key_to_misstype = key_to_misstype[:half_len]
      miss_key_direction = []
      for j in key_to_misstype:
        direction_to_shuffle = []
        if position[j][1] != 0:
          direction_to_shuffle.append([-1,0])
        if position[j][1]+1 !=  kb_shape[1]:
          direction_to_shuffle.append([1,0])
        if position[j][2] != 0:
          direction_to_shuffle.append([0,-1])
        if position[j][2]+1 !=  kb_shape[2]:
          direction_to_shuffle.append([0,1])
        random.shuffle(direction_to_shuffle)
        direction_to_misskey = direction_to_shuffle[0]
        misskey_position[j][1] += direction_to_misskey[0]
        misskey_position[j][2] += direction_to_misskey[1]
        miss_key_direction.append(direction_to_misskey)
      datasets.append({
          'word': word,
          'misskey_position': kbl.add_noise(kbl.normalize(misskey_position)),
          'correct_position': position_normalized,
          'misskey_label': misskey_position,
          'correct_label': position,
          'key_to_misstype': np.array(key_to_misstype),
          'miss_key_direction':  np.array(miss_key_direction)
      })
  return datasets


def read_wordlist(filename):
  lines = []
  with open(filename,'r',encoding='utf-8') as f:
    lines = f.readlines()
  lines = [word.strip() for word in lines if not ' ' in word]
  return lines

def main(args):
    start_timer = timer()
    wordlist = read_wordlist(args.input)
    wordlen_list = [len(w) for w in wordlist]
    shortest_len = np.min(wordlen_list)
    longest_len = np.max(wordlen_list)
    median_len = np.median(wordlen_list)
    mean_len = np.mean(wordlen_list)
    print('Total {} words'.format(len(wordlist)))
    print('Shortest word: {} characters'.format(shortest_len))
    print('Longest word: {} characters'.format(longest_len))
    print('Median word: {} characters'.format(median_len))
    print('Mean word: {:.6f} characters'.format(mean_len))
    datasets = create_dataset(wordlist,args.number)
    dataframe = pd.DataFrame(datasets)
    dataframe.to_csv(args.output)
    print('Finished in {:.2f} seconds'.format(timer() - start_timer))
    print('Output is write to {}'.format(os.path.abspath(args.output)))


def entry_point():
    parser = argparse.ArgumentParser(
        description='data_build.py - build csv training data of keyboard typing')
    parser.add_argument(
        '-i',
        '--input',
        default='data/thai-wordlist.txt',
        type=str,
        help='wordlist file (default: data/thai-wordlist.txt)',
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='data/thaiword_training_data.csv',
        help='output path of keyboard typing data csv file (default: data/thaiword_training_data.csv)')
    parser.add_argument(
        '-n',
        '--number',
        type=int,
        default=100,
        help='number of training pair per word (default: 100)')
    main(parser.parse_args())

if __name__ == "__main__":
    entry_point()