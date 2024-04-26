import os
import pandas as pd
import numpy as np
from timeshap.utils import calc_avg_event, calc_avg_sequence, get_avg_score_with_avg_event

class TimeSHAP_Explainer:
    def __init__(self, model=None, dataset:str="", use_hidden:bool=False, **kwargs):
        self.dataset = dataset
        if use_hidden:
            self.f = lambda x, y=None: model.predict_last_hs(x, y)
        else:
            self.f = lambda x: model.predict(x)

    def __call__(self, X, y, *args, **kwargs):
        save_dir = f"plots/{self.dataset}/TimeSHAP"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Iterate for each output variable (TimeSHAP assumes regression/classification output)
        for i in range(X.shape[2]):
            d_train = np.concatenate((X, y[:, :, i:i+1]), axis=2) # Shaped (n_samples, n_events, n_feats)
            model_features = list(range(d_train.shape[2]))

            average_event = [0] * d_train.shape[0]

            # Calculate average event for each sample
            for k in range(d_train.shape[0]):
                df = pd.DataFrame(d_train[k, :, :])
        
                average_event[k] = calc_avg_event(df, numerical_feats=model_features, categorical_feats=[])
                # avg_score_over_len = get_avg_score_with_avg_event(self.f, average_event[k], top=10) # tiled_background problem
            
            average_sequence = calc_avg_sequence(d_train, numerical_feats=model_features, categorical_feats=[])