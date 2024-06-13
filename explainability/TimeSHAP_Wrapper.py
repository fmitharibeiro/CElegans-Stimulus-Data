import os
import pandas as pd
import numpy as np
from altair_saver import save
from .timeshap.explainer import local_report, global_report
from .timeshap.explainer.extra import plot_background
from .timeshap import utils

class TimeSHAP_Explainer:
    def __init__(self, model=None, dataset:str="", use_hidden:bool=False, **kwargs):
        self.dataset = dataset
        self.model = model
        self.index = 0
        self.background = "median"
        self.save_dir = f"plots/{self.dataset}/TimeSHAP"
        self.local_rep = True # Compute local report?
        self.global_rep = False
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
            d_train = np.concatenate((X, dummy_descriptor), axis=2) # Shaped (n_samples, n_events, n_feats+1)
            model_features = list(range(d_train.shape[2]-1))

            if self.local_rep:
                # Calculate average event for each sample
                for k in range(d_train.shape[0]):
                    df = pd.DataFrame(d_train[k, :, :-1])
                    # avg_score_over_len = utils.get_avg_score_with_avg_event(self.f, average_event[k], top=10) # tiled_background problem

                    background = self.get_background(df, num_feats=model_features)

                    if 0 == self.index and not os.path.exists(f'{self.save_dir}/Extra/Local/background_{self.background}_seq_{k+1}.png'):
                        print(f"Background: {background.shape}")
                        plot_background(background, f'{self.save_dir}/Extra/Local/background_{self.background}_seq_{k+1}.png')
            
                    # Local Explanations (single instance)

                    # rs -> random seed, nsamples -> # of coalitions
                    pruning_dict = {'tol': 0.0001, 'path': f'{self.save_dir}/Extra/Local/prun_local_seq_{k+1}_feat_{self.index+1}.csv'} # TODO: Test tol
                    # pruning_dict = None
                    event_dict = {'rs': 33, 'nsamples': 2**15, 'path': f'{self.save_dir}/Extra/Local/event_local_seq_{k+1}_feat_{self.index+1}.csv'}
                    feature_dict = {'rs': 33, 'nsamples': 2**15, 'feature_names': model_features, 'path': f'{self.save_dir}/Extra/Local/feat_local_seq_{k+1}_feat_{self.index+1}.csv'}   #, 'plot_features': plot_feats}
                    # cell_dict = {'rs': 33, 'top_x_feats': 4, 'top_x_events': 10, 'path': f'{self.save_dir}/Extra/cell_local_seq_{k+1}_feat_{self.index+1}.csv'}
                    cell_dict = None
                    plot_report = local_report(self.f, np.expand_dims(df.to_numpy().copy(), axis=0), pruning_dict, event_dict, feature_dict, cell_dict, background, model_features=model_features, entity_col=-1, verbose=True)

                    os.makedirs(f'{self.save_dir}/Local_Reports', exist_ok=True)
                    save(plot_report, f'{self.save_dir}/Local_Reports/plot_seq_{k+1}_feat_{self.index+1}.html')
            
                    raise NotImplementedError("Testing local report")
    
            if self.global_rep:
                # average_sequence = calc_avg_sequence(d_train, numerical_feats=model_features, categorical_feats=[])
                self.background = "sequence"
                background = self.get_background(d_train, num_feats=model_features)

                schema = schema = list(model_features)
                pruning_dict = {'tol': 0.01, 'path': f'{self.save_dir}/Extra/prun_all_tf_series_{self.index+1}.csv'}
                event_dict = {'path': f'{self.save_dir}/Extra/event_all_tf_series_{self.index+1}.csv', 'rs': 42, 'nsamples': 32000}
                feature_dict = {'path': f'{self.save_dir}/Extra/feature_all_tf_series_{self.index+1}.csv', 'rs': 42, 'nsamples': 32000}
                prun_stats, global_plot = global_report(self.f, d_train, pruning_dict, event_dict, feature_dict, background, model_features, schema, entity_col=-1)
                
                # Save prun_stats to a CSV file
                prun_stats.to_csv(f'{self.save_dir}/prun_stats_{self.index+1}.csv', index=False)
                
                # Save global_plot as an HTML file using Altair
                global_plot.save(f'{self.save_dir}/global_plot_{self.index+1}.html', embed_options={'renderer': 'svg'})

            self.index += 1

    def get_background(self, X, num_feats=[], cat_feats=[]):
        if self.background == "median":
            self.background = "event"
        return getattr(utils, "calc_avg_" + self.background)(X, numerical_feats=num_feats, categorical_feats=cat_feats)

