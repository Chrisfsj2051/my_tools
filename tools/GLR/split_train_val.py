import random
import numpy as np
import os
import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data', default='data/Recognition/train.csv')
    parser.add_argument('--val_ratio', default=0.1, type=float)
    parser.add_argument('--train_out', default='data/Recognition/train_dev.txt')
    parser.add_argument('--val_out', default='data/Recognition/val_dev.txt')
    parser.add_argument('--seed', default=1, type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    data = pd.read_csv(args.raw_data)
    data = data.sample(frac=1.0)
    data = data.reset_index(drop=True)
    val_num = int(len(data) * args.val_ratio)
    val_data = data.iloc[:val_num]
    train_data = data.iloc[val_num:]
    val_data.to_csv(args.val_out, sep=' ', index=False, header=None)
    train_data.to_csv(args.train_out, sep=' ', index=False, header=None)
    # max_category_id = 203092
    # min_category_id = 1

if __name__ == '__main__':
    main()
