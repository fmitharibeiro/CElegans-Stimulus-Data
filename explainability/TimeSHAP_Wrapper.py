import os
import pandas as pd
import numpy as np
from .timeshap.utils import calc_avg_event, calc_avg_sequence, get_avg_score_with_avg_event
from .timeshap.explainer import local_report, global_report

class TimeSHAP_Explainer:
    def __init__(self, model=None, dataset:str="", use_hidden:bool=False, **kwargs):
        self.dataset = dataset
        self.model = model
        self.index = 0
        self.save_dir = f"plots/{self.dataset}/TimeSHAP"
        self.local_rep = False # Compute local report?
        if use_hidden:
            self.f = lambda x, y=None: self.model.predict_last_hs(x, y)[:, :, self.index]
        else:
            self.f = lambda x: self.model.predict(x)[:, :, self.index]

    def __call__(self, X, y, *args, **kwargs):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # Iterate for each output variable (TimeSHAP assumes regression/classification output)
        for _ in range(X.shape[2]):
            # d_train = np.concatenate((X, y[:, :, i:i+1]), axis=2) # Shaped (n_samples, n_events, n_feats)
            dummy_descriptor = np.zeros((X.shape[0], X.shape[1], 1))  # Creating a dummy descriptor column filled with zeros
            d_train = np.concatenate((X, dummy_descriptor), axis=2) # Shaped (n_samples, n_events, n_feats)
            model_features = list(range(d_train.shape[2]-1))

            average_event = [0] * d_train.shape[0]

            if self.local_rep:
                # Calculate average event for each sample
                for k in range(d_train.shape[0]):
                    df = pd.DataFrame(d_train[k, :, :-1])
            
                    average_event[k] = calc_avg_event(df, numerical_feats=model_features, categorical_feats=[])
                    # avg_score_over_len = get_avg_score_with_avg_event(self.f, average_event[k], top=10) # tiled_background problem
            
                    # Local Explanations (single instance)

                    # rs -> random seed, nsamples -> # of coalitions
                    pruning_dict = {'tol': 0.025} # TODO: Test tol
                    # pruning_dict = None
                    event_dict = {'rs': 33, 'nsamples': 32000}
                    feature_dict = {'rs': 33, 'nsamples': 32000, 'feature_names': model_features}   #, 'plot_features': plot_feats}
                    cell_dict = {'rs': 33, 'nsamples': 32000, 'top_x_feats': 4, 'top_x_events': 4}
                    local_report(self.f, np.expand_dims(df.to_numpy().copy(), axis=0), pruning_dict, event_dict, feature_dict, cell_dict, average_event[k], model_features=model_features) #entity_col?
    
            average_sequence = calc_avg_sequence(d_train, numerical_feats=model_features, categorical_feats=[])

            schema = schema = list(model_features)
            pruning_dict = {'tol': 0.01, 'path': f'{self.save_dir}/prun_all_tf_series_{self.index+1}.csv'}
            event_dict = {'path': f'{self.save_dir}/event_all_tf_series_{self.index+1}.csv', 'rs': 42, 'nsamples': 32000}
            feature_dict = {'path': f'{self.save_dir}/feature_all_tf_series_{self.index+1}.csv', 'rs': 42, 'nsamples': 32000}
            prun_stats, global_plot = global_report(self.f, d_train, pruning_dict, event_dict, feature_dict, average_sequence, model_features, schema, entity_col=-1)
            print(prun_stats) # TODO: Understand output prints
            print(global_plot)

            self.index += 1


