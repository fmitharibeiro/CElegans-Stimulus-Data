import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
import optuna
from functools import partial


class CustomCV():
    '''
    Custom Hyperparameter search with TPE
    '''

    def __init__(self, estimator, param_distributions, n_trials, seed, *args, **kwargs):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.cv = None
        self.seed = seed
        self.n_trials = n_trials
       
    def fit(self, X, y, *args, **kwargs):

        # For safety, normalize both X and y to numpy arrays if they are pandas datafarmes
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        # Hyperparameter search based on MSE
        def objective(trial, X_aux, y_aux, param_grid, cv_object, estimator):
            params = {param: getattr(trial, value[0])(param, *value[1:]) for (param, value) in param_grid.items()}
            scores = []

            for i, (train_index, test_index) in enumerate(cv_object):

                X_train, y_train = X_aux[train_index], y_aux[train_index]
                X_test, y_test = X_aux[test_index], y_aux[test_index]

                estimator.set_params(**params)
                estimator.fit(X_train, y_train)

                preds = estimator.predict(X_test)
                score = mean_squared_error(y_test, preds)
                scores.append(score)
                print(f"MSE obtained on {i}-th fold: {score}")

            return np.mean(scores)
        
        # Perform 5-fold cross validation
        skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = self.seed)
        self.cv = [idx for idx in skf.split(X, y)]
    

        clf_objective = partial(objective, X_aux = X, y_aux = y, param_grid = self.param_distributions, 
                                cv_object = self.cv, estimator = self.estimator)
            
        print(f"Finding best hyperparameter combinations...")
        print(self.estimator.get_params())
        study = optuna.create_study(direction='minimize', sampler = optuna.samplers.TPESampler(seed = self.seed))
        #study.enqueue_trial({key: value for (key, value) in self.estimator.get_params().items() \
        #                    if key in self.param_distributions.keys()})
        study.optimize(clf_objective, n_trials = self.n_trials)

        self.best_score_ = study.best_value
        self.best_params_ = study.best_params

        print("Best parameter for classifier (CV score=%0.3f):" %  self.best_score_)
        print(f"{self.best_params_}")

        # Refit the estimator to the best hyperparameters
        self.best_estimator_ = self.estimator
        for (attr, value) in self.best_params_.items():
            self.best_estimator_.set_params(**{attr: value})

        return self.best_estimator_