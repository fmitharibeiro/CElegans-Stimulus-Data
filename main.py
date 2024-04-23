import os
import sys
import time
import argparse
import numpy as np
import torch
import tensorflow
import optuna

from CustomCV import CustomCV
import utils
import methods


def main(opt):
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    tensorflow.random.set_seed(opt.seed)

    data = utils.fetch_data(opt.dataset)
    X_train, y_train = data["train"]
    X_test, y_test = data["test"]

    name = ""
    param_grid = {}

    met = methods.fetch_method(opt.method, opt.seed)
    name += f"{opt.method}"

    for (parameter, values) in met.param_grid.items():
        param_grid[str(parameter)] = values

    # To perform post-hoc methods, we must first have a defined classifier
    if name == "TimeSHAP" and os.path.exists(f"config/{opt.dataset}/Base{opt.dataset}.db"):
        print(f"Fetching base model best configuration...")
        # Load the study from the SQLite database
        study = optuna.load_study(
            study_name=f'{opt.dataset}:{name}-study',
            storage=f'sqlite:///config/{opt.dataset}/{name}.db'
        )
        # Get the best hyperparameters
        best_params = study.best_params
        
        met.set_params(**{key: value for (key, value) in best_params})
        print(f"After fetching base classifier it had parameters: {met.params}")
    elif name == "TimeSHAP":
        print(f"Base model best configuration not found. Train base model first. (Base{opt.dataset})")
        sys.exit()

    start_time = time.time()

    search = CustomCV(estimator = met, 
                      param_distributions = param_grid, 
                      n_trials = opt.n_trials,
                      seed = opt.seed,
                      name = f"{opt.dataset}:{opt.method}"
                      )
    
    if name == "BaseCE":
        est = search.fit(X_train, y_train)
    else:
        # TODO: y_train for each variable
        est = search.fit(X_train, y_train[:, :, 0])
    total_time = time.time() - start_time

    print(f"Total time: {total_time}")

    # Print output




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['CE'], default='CE', help='Dataset to run')
    parser.add_argument('--method', type=str, choices=['IMV-LSTM', 'TimeSHAP', 'BaseCE'], default=None, help='Explainable method to run')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_trials', type=int, default=50, help='Number of optimization trials to run')
    opt = parser.parse_args()
    
    main(opt)