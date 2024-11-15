import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import optuna
from functools import partial


class CustomCV():
    '''
    Custom Hyperparameter search with TPE
    '''

    def __init__(self, estimator, param_distributions, n_trials, seed, name, *args, **kwargs):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.cv = None
        self.seed = seed
        self.name = name
        self.n_trials = n_trials
        self.skip_train = kwargs.get("skip_train", False)
       
    def fit(self, X, y, *args, **kwargs):

        # For safety, normalize both X and y to numpy arrays if they are pandas dataframes
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        
        # Create "config" directory
        if not os.path.exists(f"config/{self.name.split(':')[0]}"):
            os.makedirs(f"config/{self.name.split(':')[0]}")

        # Hyperparameter search based on normalized MSE and MAE
        def objective(trial, X_aux, y_aux, param_grid, cv_object, estimator):
            params = {param: getattr(trial, value[0])(param, *value[1:]) for (param, value) in param_grid.items()}
            estimator.set_params(**params)
            print(f"Trying {params}")

            rmse_scores = []
            mae_scores = []

            for i, (train_index, test_index) in enumerate(cv_object):
                X_train, y_train = X_aux[train_index], y_aux[train_index]
                X_test, y_test = X_aux[test_index], y_aux[test_index]

                estimator.fit(X_train, y_train)

                preds = estimator.predict(X_test)

                print(f"preds, y_test shapes: {preds.shape}, {y_test.shape}")

                if preds.ndim > 2:
                    for j in range(preds.shape[2]):
                        mse_score = mean_squared_error(y_test[:, :, j], preds[:, :, j])
                        mae_score = mean_absolute_error(y_test[:, :, j], preds[:, :, j])
                        rmse_score = np.sqrt(mse_score)
                        rmse_scores.append(rmse_score)
                        mae_scores.append(mae_score)
                        print(f"RMSE, MAE obtained on fold {i+1}, series {j+1}: {rmse_score}, {mae_score}")
                else:
                    mse_score = mean_squared_error(y_test, preds)
                    mae_score = mean_absolute_error(y_test, preds)
                    rmse_score = np.sqrt(mse_score)
                    rmse_scores.append(rmse_score)
                    mae_scores.append(mae_score)
                    print(f"RMSE, MAE obtained on fold {i+1}: {rmse_score}, {mae_score}")

            weight = 0.9

            # Calculate the combined score using the mean MSE and MAE
            combined_score = weight * np.mean(rmse_scores) + (1 - weight) * np.mean(mae_scores)

            return combined_score
        
        print(f"X: {X.shape}")
        print(f"y: {y.shape}")
        
        # Perform 3-fold cross validation
        kf = KFold(n_splits=3, shuffle=True, random_state=self.seed)
        self.cv = list(kf.split(X, y))

        clf_objective = partial(objective, X_aux = X, y_aux = y, param_grid = self.param_distributions, 
                                cv_object = self.cv, estimator = self.estimator)
            
        print(f"Finding best hyperparameter combinations...")
        print(self.estimator.param_grid)
        
        study = optuna.create_study(study_name=f'{self.name}-study',
                        direction='minimize',
                        sampler=optuna.samplers.TPESampler(),
                        storage=f'sqlite:///config/{self.name.split(":")[0]}/{self.name.split(":")[1]}.db',
                        load_if_exists=True)
        
        if not self.skip_train:
            study.optimize(clf_objective, n_trials = self.n_trials)

        self.best_score_ = study.best_value
        self.best_params_ = study.best_params

        print("Best parameter for classifier (CV score=%0.3f):" %  self.best_score_)
        print(f"{self.best_params_}")

        # Refit the estimator to the best hyperparameters
        self.best_estimator_ = self.estimator
        for (attr, value) in self.best_params_.items():
            self.best_estimator_.set_params(**{attr: value})
        
        self.best_estimator_.fit(X, y)

        return self.best_estimator_