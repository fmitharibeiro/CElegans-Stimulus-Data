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

    data = utils.fetch_data(opt.dataset, opt.reduce, other_args=opt)
    X_train, y_train = data["train"]
    X_test, y_test = data["test"]

    opt.input_size = X_train.shape[2]
    opt.output_size = X_test.shape[2]
    opt.base_name = f"Base{opt.dataset}_Torch" if opt.torch else f"Base{opt.dataset}"

    name = ""
    param_grid = {}

    met = methods.fetch_method(opt.method, opt.seed, other_args=opt)
    name += opt.method[:-6] if opt.method.endswith("_Torch") else opt.method

    if not os.path.exists(f"plots/{opt.dataset}/Data"):
        os.makedirs(f"plots/{opt.dataset}/Data")

        utils.plot_all_samples(np.concatenate((X_train, X_test), axis=0), f"plots/{opt.dataset}/Data", 'inputs.png')
        utils.plot_all_samples(np.concatenate((y_train, y_test), axis=0), f"plots/{opt.dataset}/Data", 'outputs.png')

    if met:
        for (parameter, values) in met.param_grid.items():
            param_grid[str(parameter)] = values

    start_time = time.time()

    # To perform post-hoc methods, we must first have a defined classifier
    if name in ["TimeSHAP", "SeqSHAP"] and os.path.exists(f"config/{opt.dataset}/{opt.base_name}_{opt.num_hidden_layers}.db"):
        print(f"Fetching base model best configuration...")

        # Load the study from the SQLite database
        try:
            study = optuna.load_study(
                study_name=f'{opt.dataset}:{opt.base_name}_{opt.num_hidden_layers}-study',
                storage=f'sqlite:///config/{opt.dataset}/{opt.base_name}_{opt.num_hidden_layers}.db'
            )
        # TODO: Study name bug, to be removed
        except KeyError:
            study = optuna.load_study(
                study_name=f'{opt.dataset}:{opt.base_name}-study',
                storage=f'sqlite:///config/{opt.dataset}/{opt.base_name}_{opt.num_hidden_layers}.db'
            )
        # Get the model
        base_model = methods.fetch_method(opt.base_name, opt.seed, other_args=opt)

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
            utils.plot_predictions(base_model, X_test, y_test, save_dir=f"plots/{opt.dataset}/{opt.method}/BaseModel")

            utils.print_metrics(base_model, X_test, y_test, start_time, save_dir=f"plots/{opt.dataset}/{opt.method}/BaseModel")

        if met:
            met.set_params(**{"model": base_model})

        out = explainability.fetch_explainer(opt.method, model=base_model, dataset=opt.dataset, use_hidden=opt.torch, seed=opt.seed, other_args=opt)

        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)

        # # Calculate mean and standard deviation along the features axis for each sample
        # mean = np.mean(X, axis=1, keepdims=True)
        # std = np.std(X, axis=1, keepdims=True)

        # # Normalize dataset along the features axis
        # X_normalized = (X - mean) / std
        # print(f"X_normalized: {X_normalized[0, :, 0]}")

        out(X, y)


    elif name in ["TimeSHAP", "SeqSHAP"]:
        print(f"Base model best configuration not found. Train base model first. ({opt.method}_{opt.num_hidden_layers})")
        sys.exit()
    else:
        search = CustomCV(estimator = met, 
                        param_distributions = param_grid, 
                        n_trials = opt.n_trials,
                        seed = opt.seed,
                        name = f"{opt.dataset}:{opt.method}_{opt.num_hidden_layers}",
                        skip_train = opt.skip_train
                        )
        
        if name in ["BaseCE"]:
            est = search.fit(X_train, y_train)
        else:
            # TODO: y_train for each variable
            est = search.fit(X_train, y_train[:, :, 0])

        if opt.plot:
            # Use utils to check if model fitted well to data
            utils.plot_predictions(est, X_test, y_test, save_dir=f"plots/Base_Models/{opt.method}_{opt.num_hidden_layers}")

            # Check training outputs
            utils.plot_predictions(est, X_train, y_train, save_dir=f"plots/Base_Models/{opt.method}_{opt.num_hidden_layers}/Train")

            utils.print_metrics(est, X_test, y_test, start_time, save_dir=f"plots/Base_Models/{opt.method}_{opt.num_hidden_layers}")

        utils.save_model(est, opt.dataset)

    total_time = time.time() - start_time

    print(f"Total time: {total_time}")







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['CE'], default='CE', help='Dataset to run')
    parser.add_argument('--method', type=str, choices=['IMV-LSTM', 'TimeSHAP', 'SeqSHAP', 'BaseCE'], default=None, help='Explainable method to run')
    parser.add_argument('--reduce', type=float, default=1., help='Reduce dataset (between 0.0 and 1.0). A value of 0.25 means 1/4 of dataset used.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--plot', action='store_false', help='Save plots?')
    parser.add_argument('--torch', action='store_true', help='Use PyTorch?')
    # Model training only
    parser.add_argument('--num_hidden_layers', type=int, default=8, help='Number of base model hidden layers')
    # parser.add_argument('--output_size', type=int, default=4, help='Number of outputs (1 for each output series)')
    parser.add_argument('--skip_train', action='store_true', help='Skips the training and fits directly the best model. In TimeSHAP, uses current saved data only.')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of optimization trials to run')
    # TimeSHAP only
    parser.add_argument('--no_local', action='store_false', help='TimeSHAP only. Compute local reports?')
    parser.add_argument('--no_global', action='store_false', help='TimeSHAP only. Compute global reports?')
    parser.add_argument('--verbose', action='store_true', help='TimeSHAP only. Turn on verbose?')
    opt = parser.parse_args()
    
    assert opt.method is not None
    opt.method = opt.method + "_Torch" if opt.torch else opt.method
    main(opt)