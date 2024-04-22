import os
import time
import argparse
import numpy as np
import torch

from CustomCV import CustomCV
import utils
import methods


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = utils.fetch_data(args.dataset)
    X_train, y_train = data["train"]
    X_test, y_test = data["test"]

    name = ""
    param_grid = {}

    met = methods.fetch_method(args.method, args.seed)
    name += f"{args.method}"

    for (parameter, values) in met.param_grid.items():
        param_grid[str(parameter)] = values

    start_time = time.time()

    search = CustomCV(estimator = met, 
                      param_distributions = param_grid, 
                      n_trials = args.n_trials,
                      seed = args.seed,
                      name = f"{args.dataset}:{args.method}"
                      )
    
    # TODO: y_train for each variable
    est = search.fit(X_train, y_train[:, :, 0])
    total_time = time.time() - start_time

    print(f"Total time: {total_time}")

    # Print output




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['CE'], default='CE', help='Dataset to run')
    parser.add_argument('--method', type=str, choices=['IMV-LSTM'], default='IMV-LSTM', help='Explainable method to run')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_trials', type=int, default=50, help='Number of optimization trials to run')
    args = parser.parse_args()
    
    main(args)