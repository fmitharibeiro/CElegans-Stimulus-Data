import os
import argparse

import utils


def main(args):
    data = utils.fetch_data(args.dataset)
    X_train, y_train = data["train"]
    X_test, y_test = data["test"]

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['CE'], default='CE', help='Dataset to run')
    args = parser.parse_args()
    
    main(args)