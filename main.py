import os
import argparse

import utils
import methods


def main(args):
    data = utils.fetch_data(args.dataset)
    X_train, y_train = data["train"]
    X_test, y_test = data["test"]

    name = ""
    steps = []
    param_grid = {}

    met = methods.fetch_method(args.method)
    name += f"_{args.method}"

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['CE'], default='CE', help='Dataset to run')
    parser.add_argument('--method', type=str, choices=['IMV-LSTM'], default='IMV-LSTM', help='Explainable method to run')
    args = parser.parse_args()
    
    main(args)