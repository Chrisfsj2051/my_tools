import random

import mmcv
import numpy as np
import os
import pandas as pd
import argparse

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/Recognition/test/')
    parser.add_argument('--ann_out', default='data/generated_anns/Recognition/test_dev.txt')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    filename_list = []
    for root, folders, files in tqdm(os.walk(args.data_path)):
        filename_list += files
    filename_list = [x.replace('.jpg', '\n') for x in filename_list]
    with open(args.ann_out, 'w') as f:
        f.writelines(filename_list)
    print(f'Saved as {args.ann_out}')


if __name__ == '__main__':
    main()
