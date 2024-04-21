import os
import numpy as np
import pandas as pd
from scipy.io import loadmat


class CEHandler():
    def __init__(self):
        self.dir_name = 'CE'
        self.test_ratio = 0.2
    
    def fetch_data(self):

        if not os.path.exists(f"datasets/{self.dir_name}/features/features_train.npy"):

            mat_contents = loadmat(f'datasets/{self.dir_name}/Sequences40.mat')
            times = mat_contents['time']
            inputs = mat_contents['ii']
            outputs = mat_contents['oo']

            num_samples = inputs.shape[2]
            num_time_series = inputs.shape[1]

            df = np.zeros((num_samples, len(times), num_time_series*2))

            # Generate columns for each time series
            for j in range(num_samples):
                for i in range(num_time_series*2):
                    if i < num_time_series:
                        time_series = [inputs[k, i, j] for k in range(len(times))]
                    else:
                        time_series = [outputs[k, i-num_time_series, j] for k in range(len(times))]
                    df[j, :, i] = time_series

            # Split data into train and test sets
            num_test_samples = int(self.test_ratio * num_samples)
            train_data = df[:-num_test_samples]
            test_data = df[-num_test_samples:]

            X_train = train_data[:, :, :num_time_series]
            y_train = train_data[:, :, num_time_series:]
            X_test = test_data[:, :, :num_time_series]
            y_test = test_data[:, :, num_time_series:]

            if not os.path.exists(f"datasets/{self.dir_name}/features"):
                os.makedirs(f"datasets/{self.dir_name}/features")

            np.save(f"datasets/{self.dir_name}/features/features_train.npy", X_train)
            np.save(f"datasets/{self.dir_name}/features/targets_train.npy", y_train)
            np.save(f"datasets/{self.dir_name}/features/features_test.npy", X_test)
            np.save(f"datasets/{self.dir_name}/features/targets_test.npy", y_test)
        
        else:
            X_train = np.load(f"datasets/{self.dir_name}/features/features_train.npy")
            y_train = np.load(f"datasets/{self.dir_name}/features/targets_train.npy")
            X_test = np.load(f"datasets/{self.dir_name}/features/features_test.npy")
            y_test = np.load(f"datasets/{self.dir_name}/features/targets_test.npy")
        
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")

        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")

        return X_train, y_train, X_test, y_test