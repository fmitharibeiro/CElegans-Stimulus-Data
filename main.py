import os
import sys
import time
import argparse
import numpy as np
import torch
import tensorflow
import optuna
import random

from CustomCV import CustomCV
import utils
import methods
import explainability

from sklearn.metrics import mean_squared_error


def main(opt):
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    tensorflow.random.set_seed(opt.seed)

    data = utils.fetch_data(opt.dataset, opt.reduce)
    X_train, y_train = data["train"]
    X_test, y_test = data["test"]

    name = ""
    param_grid = {}

    met = methods.fetch_method(opt.method, opt.seed)
    name += f"{opt.method}"

    if met:
        for (parameter, values) in met.param_grid.items():
            param_grid[str(parameter)] = values

    start_time = time.time()

    # To perform post-hoc methods, we must first have a defined classifier
    if name in ["TimeSHAP", "SeqSHAP"] and os.path.exists(f"config/{opt.dataset}/Base{opt.dataset}.db"):
        print(f"Fetching base model best configuration...")

        # Load the study from the SQLite database
        study = optuna.load_study(
            study_name=f'{opt.dataset}:Base{opt.dataset}-study',
            storage=f'sqlite:///config/{opt.dataset}/Base{opt.dataset}.db'
        )
        # Get the model
        base_model = methods.fetch_method(f"Base{opt.dataset}", opt.seed)

        # Get the best hyperparameters (e.g. BaseCE prediction needs batch_size)
        best_params = study.best_params
        print(f"Best params: {best_params}")

        base_model.set_params(**{key: value for (key, value) in best_params.items()})
        print(f"After fetching base classifier, it had parameters: {base_model.get_params()}")

        base_model_model = utils.load_model(opt.dataset)
        base_model.set_params(**{"model": base_model_model})

        # Print normalized MSE (max-min)
        preds = base_model.predict(X_test)
        for i in range(y_test.shape[2]):
            mse = mean_squared_error(y_test[:, :, i], preds[:, :, i])
            max_true = np.max(y_test[:, :, i])
            min_true = np.min(y_test[:, :, i])
            range_true = max_true - min_true
            normalized_mse = mse / range_true
            print(f"Base model normalized MSE, series {i+1}: {normalized_mse}")
        
        if opt.plot:
            # Use utils to check if model fitted well to data
            utils.plot_predictions(base_model, X_test, y_test, save_dir=f"plots/{opt.dataset}/{name}/BaseModel")

            utils.print_metrics(base_model, X_test, y_test, start_time, save_dir=f"plots/{opt.dataset}/{name}/BaseModel")

        if met:
            met.set_params(**{"model": base_model})
            out = explainability.fetch_explainer(opt.method, model=met, dataset=opt.dataset, use_hidden=True, seed=opt.seed)
        else:
            out = explainability.fetch_explainer(opt.method, model=base_model, dataset=opt.dataset, use_hidden=False, seed=opt.seed)
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        out(X, y)


    elif name in ["TimeSHAP", "SeqSHAP"]:
        print(f"Base model best configuration not found. Train base model first. (Base{opt.dataset})")
        sys.exit()
    else:
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

        if opt.plot:
            # Use utils to check if model fitted well to data
            utils.plot_predictions(est, X_test, y_test, save_dir=f"plots/Base_Models/{name}")

            utils.print_metrics(est, X_test, y_test, start_time, save_dir=f"plots/Base_Models/{name}")

        utils.save_model(est, opt.dataset)

    total_time = time.time() - start_time

    print(f"Total time: {total_time}")







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['CE'], default='CE', help='Dataset to run')
    parser.add_argument('--reduce', type=float, default=1., help='Reduce dataset (between 0.0 and 1.0)')
    parser.add_argument('--method', type=str, choices=['IMV-LSTM', 'TimeSHAP', 'SeqSHAP', 'BaseCE'], default=None, help='Explainable method to run')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--plot', type=bool, default=True, help='Save plots?')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of optimization trials to run')
    opt = parser.parse_args()
    
    assert opt.method is not None
    main(opt)