import os
import numpy as np
import tensorflow as tf
from scipy.io import loadmat


class CEHandler():
    def __init__(self):
        self.dir_name = 'CE'
        self.test_ratio = 0.2
        self.train_test_split = 'manual' # 'manual' / 'threshold'
        self.test_indices = [8, 10, 19, 22, 23, 29, 38, 40] # For 'manual' only, starts at 1 (not 0)
        self.model_file = f"datasets/{self.dir_name}/features/BaseCE.h5"
    
    def fetch_data(self):

        if not os.path.exists(f"datasets/{self.dir_name}/features/features_train.npy"):

            mat_contents = loadmat(f'datasets/{self.dir_name}/Sequences40.mat')
            times = mat_contents['time']
            inputs = mat_contents['ii']
            outputs = mat_contents['oo']

            num_samples = inputs.shape[2]
            num_time_series = inputs.shape[1]

            X = np.zeros((num_samples, len(times), num_time_series*2))

            # Generate columns for each time series
            for j in range(num_samples):
                for i in range(num_time_series*2):
                    if i < num_time_series:
                        time_series = [inputs[k, i, j] for k in range(len(times))]
                    else:
                        time_series = [outputs[k, i-num_time_series, j] for k in range(len(times))]
                    X[j, :, i] = time_series

            X_train, y_train, X_test, y_test = self.data_split(X, num_samples=num_samples, num_time_series=num_time_series)

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

    def data_split(self, X, num_samples=None, num_time_series=None):
        if self.train_test_split == 'threshold':
            # Split data into train and test sets
            num_test_samples = int(self.test_ratio * num_samples)
            train_data = X[:-num_test_samples]
            test_data = X[-num_test_samples:]
        
        elif self.train_test_split == 'manual':
            test_indices = np.array(self.test_indices)-1

            # Create a mask for the test indices
            mask = np.ones(X.shape[0], dtype=bool)
            mask[test_indices] = False
            
            # Split the data into training and testing sets
            train_data = X[mask]
            test_data = X[test_indices]
        
        else:
            raise NotImplementedError("Only threshold and manual train-test splits are implemented")
        
        X_train = train_data[:, :, :num_time_series]
        y_train = train_data[:, :, num_time_series:]
        X_test = test_data[:, :, :num_time_series]
        y_test = test_data[:, :, num_time_series:]

        return X_train, y_train, X_test, y_test
    
    def save_model(self, model):
        model.save(self.model_file)
        print(f"Model saved successfully in {self.model_file}.")
    
    def load_model(self):
        try:
            loaded_model = tf.keras.models.load_model(self.model_file)
            print("Model loaded successfully.")
            return loaded_model
        except FileNotFoundError:
            raise FileNotFoundError("Specified file does not exist.")
        except ValueError as e:
            raise ValueError("Error loading the model: {}".format(str(e)))