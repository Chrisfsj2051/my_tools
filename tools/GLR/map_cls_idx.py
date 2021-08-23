import random

import mmcv
import numpy as np
import os
import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data', default='data/Recognition/train.csv')
    parser.add_argument('--out', default='data/generated_anns/Recognition/')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data = pd.read_csv(args.raw_data)
    unique_ids = data['landmark_id'].unique()
    np.save('configs/unique_landmark_ids.np', unique_ids)

if __name__ == '__main__':
    main()
