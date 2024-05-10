import os, time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from datasets import CEHandler


datasets = {"CE": CEHandler()}

def fetch_data(dataset, reduct):
    """
    Loads the dataset, performing a 80-20 train-test split
    """
    if not dataset in datasets:
        raise AssertionError
    assert 0 <= reduct <= 1, "Reduction factor must be between 0 and 1"

    handler = datasets[dataset]
    X_train, y_train, X_test, y_test = handler.fetch_data()

    red_ind_train = int(X_train.shape[0] * reduct)
    red_ind_test = int(X_test.shape[0] * reduct)
    X_train = X_train[:red_ind_train]
    y_train = y_train[:red_ind_train]
    X_test = X_test[:red_ind_test]
    y_test = y_test[:red_ind_test]

    print(f"Dataset has {red_ind_train} training and {red_ind_test} test samples.")

    return {'train':(X_train, y_train), 'test':(X_test, y_test)}


def save_model(model, dataset):
    """
    Save a model to a file, specific to dataset.
    """
    handler = datasets[dataset]
    handler.save_model(model)
    

def load_model(dataset):
    """
    Load a model from a dataset.
    """
    handler = datasets[dataset]
    return handler.load_model()


def plot_predictions(model, X, y_true, save_dir="plots"):
    """
    Plots model predictions against ground truth values for each sample divided by features.
    
    Parameters:
        model: Model object with a predict method.
        X (numpy.ndarray): Input data shaped (n_samples, times, feat).
        y_true (numpy.ndarray): Ground truth values shaped (n_samples, times, feat).
        save_dir (str): Directory to save the plots. Default is "plots".
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    n_samples, _, feats = X.shape
    
    for j in range(feats):
        fig, axes = plt.subplots(n_samples, 1, figsize=(10, 5 * n_samples))
        
        for i in range(n_samples):
            y_pred = model.predict(X[i:i+1])
            axes[i].plot(y_true[i, :, j], label='Ground Truth')
            axes[i].plot(y_pred[0, :, j], label='Predictions')
            axes[i].set_title(f'Sample {i+1}')
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'predictions_feature_{j+1}.png'))
        plt.close()


def print_metrics(model, X_test, y_test, start_time, save_dir="metrics"):
    """
    Prints evaluation metrics for each sample and feature to a .txt file.
    
    Parameters:
        model: Model object with a predict method.
        X_test (numpy.ndarray): Input test data shaped (n_samples, times, feat).
        y_test (numpy.ndarray): Ground truth test values shaped (n_samples, times, feat).
        start_time (float): Time when the evaluation started.
        save_dir (str): Directory to save the .txt file. Default is "metrics".
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    n_samples, _, feat = X_test.shape
    preds = model.predict(X_test)
    
    with open(os.path.join(save_dir, "evaluation_metrics.txt"), "w") as f:
        for i in range(n_samples):
            f.write(f"Sample {i+1}:\n")
            for j in range(feat):
                mse = mean_squared_error(y_test[i, :, j], preds[i, :, j])
                mean_true = np.mean(y_test[i, :, j])
                variance_true = np.var(y_test[i, :, j])
                max_true = np.max(y_test[i, :, j])
                min_true = np.min(y_test[i, :, j])
                range_true = max_true - min_true
                
                normalized_mse_range = mse / range_true
                normalized_mse_variance = mse / variance_true
                
                f.write(f"Feature {j+1}:\n")
                f.write(f"Normalized MSE (Range): {normalized_mse_range}\n")
                f.write(f"Normalized MSE (Variance): {normalized_mse_variance}\n")
                f.write(f"Mean: {mean_true}\n")
                f.write(f"Variance: {variance_true}\n")
                f.write(f"Range: {range_true}\n\n")
    
    total_time = time.time() - start_time
    formatted_total_time = "{:.2f}".format(total_time)
    with open(os.path.join(save_dir, "evaluation_metrics.txt"), "a") as f:
        f.write(f"Total Time: {formatted_total_time} seconds\n")
