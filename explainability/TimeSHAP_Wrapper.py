import os
import pandas as pd
import numpy as np
from altair_saver import save
from .timeshap.explainer import local_report, global_report
from .timeshap.explainer.extra import plot_background
from .timeshap import utils
from .timeshap.wrappers import TorchModelWrapper, TensorFlowModelWrapper

class TimeSHAP_Explainer:
    def __init__(self, model=None, dataset:str="", use_hidden:bool=False, **kwargs):
        self.dataset = dataset
        self.model = model
        self.index = 0
        self.background = "median"
        self.tol = 0.001
        self.seed = 33
        self.nsamples = 2**15
        self.downsample_rate = 50
        self.save_dir = f"plots/{self.dataset}/TimeSHAP"
        self.save_dir_pruning = self.save_dir
        self.local_rep = getattr(kwargs.get('other_args'), 'no_local')
        self.global_rep = getattr(kwargs.get('other_args'), 'no_global')
        self.compute_cell = getattr(kwargs.get('other_args'), 'no_cell')
        self.verbose = getattr(kwargs.get('other_args'), 'verbose')
        self.skip_train = getattr(kwargs.get('other_args'), 'skip_train')
        self.torch = getattr(kwargs.get('other_args'), 'torch')

        if use_hidden and self.torch:
            # Torch model
            self.save_dir += "_Torch"

            model_wrapped = TorchModelWrapper(self.model, batch_budget=self.nsamples, batch_ignore_seq_len=True)
            self.f = lambda x, y=None: model_wrapped.predict_last_hs(x, y, return_hidden=True)[:, :, self.index]
        elif use_hidden:
            # TensorFlow model w/ hidden state
            self.save_dir += "_hidden"

            model_wrapped = TensorFlowModelWrapper(self.model, batch_budget=self.nsamples, batch_ignore_seq_len=True)
            self.f = lambda x, y=None: model_wrapped.predict_last_hs(x, y, return_hidden=True, index=self.index)
        else:
            # TensorFlow model without hidden state
            self.f = lambda x: self.model.predict(x, verbose=self.verbose)[:, :, self.index]

    def __call__(self, X, y, *args, **kwargs):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # Iterate for each output variable (TimeSHAP assumes regression/classification output)
        for _ in range(X.shape[2]):
            dummy_descriptor = np.zeros((X.shape[0], X.shape[1], 1))
            d_train = np.concatenate((X, dummy_descriptor), axis=2)
            model_features = list(range(d_train.shape[2]-1))

            if self.local_rep:
                for k in range(d_train.shape[0]):
                    # Local Explanations (single instance)
                    df = pd.DataFrame(d_train[k, :, :-1])

                    self.background = "median"
                    background = self.get_background(df, num_feats=model_features)

                    if 0 == self.index and not os.path.exists(f'{self.save_dir}/Extra/Local/Sequence_{k+1}/background_{self.background}.png'):
                        print(f"Background: {background.shape}")
                        plot_background(background, (d_train.shape[1], d_train.shape[2]-1), f'{self.save_dir}/Extra/Local/Sequence_{k+1}/background_{self.background}.png')
            
                    # rs -> random seed, nsamples -> nr of coalitions, tol -> tolerance (%)
                    pruning_dict = {'tol': self.tol, 'path': f'{self.save_dir_pruning}/Extra/Local/Sequence_{k+1}/Feature_{self.index+1}/prun_local.csv'}
                    event_dict = {'rs': self.seed, 'nsamples': self.nsamples, 'path': f'{self.save_dir}/Extra/Local/Sequence_{k+1}/Feature_{self.index+1}/event_local.csv'}
                    feature_dict = {'rs': self.seed, 'nsamples': self.nsamples, 'path': f'{self.save_dir}/Extra/Local/Sequence_{k+1}/Feature_{self.index+1}/feat_local.csv'}
                    cell_dict = {'rs': self.seed, 'top_x_feats': 4, 'top_x_events': 1000, 'path': f'{self.save_dir}/Extra/Local/Sequence_{k+1}/Feature_{self.index+1}/cell_local.csv'}
                    cell_dict = cell_dict if self.compute_cell else None
                    plot_report = local_report(self.f, np.expand_dims(df.to_numpy().copy(), axis=0), pruning_dict, event_dict, feature_dict, cell_dict, background, model_features=model_features, entity_col=-1, verbose=self.verbose)

                    os.makedirs(f'{self.save_dir}/Local_Reports/Sequence_{k+1}', exist_ok=True)
                    save(plot_report, f'{self.save_dir}/Local_Reports/Sequence_{k+1}/plot_seq_feat_{self.index+1}.html')

            
            if self.global_rep:
                # Global Explanations (single output feature)
                self.background = "sequence"
                background = self.get_background(d_train[:, :, :-1], num_feats=model_features)

                if 0 == self.index and not os.path.exists(f'{self.save_dir}/Extra/Global/background_{self.background}.png'):
                    print(f"Background: {background.shape}")
                    plot_background(background, (d_train.shape[1], d_train.shape[2]-1), f'{self.save_dir}/Extra/Global/background_{self.background}.png')

                schema = list(model_features)
                pruning_dict = {'tol': self.tol, 'path': f'{self.save_dir_pruning}/Extra/Global/Feature_{self.index+1}/prun_global.csv'}
                event_dict = {'rs': self.seed, 'nsamples': self.nsamples, 'path': f'{self.save_dir}/Extra/Global/Feature_{self.index+1}/event_global.csv', 'skip_train': self.skip_train, 'num_outputs': X.shape[1], 'downsample_rate': self.downsample_rate}
                feature_dict = {'rs': self.seed, 'nsamples': self.nsamples, 'path': f'{self.save_dir}/Extra/Global/Feature_{self.index+1}/feature_global_feat.csv', 'skip_train': self.skip_train, 'num_outputs': X.shape[1], 'downsample_rate': self.downsample_rate}
                prun_stats, global_plot = global_report(self.f, d_train, pruning_dict, event_dict, feature_dict, background, model_features, schema, entity_col=-1, verbose=self.verbose)
                
                if prun_stats is not None:
                    prun_stats.to_csv(f'{self.save_dir}/prun_stats_feat_{self.index+1}.csv', index=False)
                
                if global_plot is not None:
                    global_plot.save(f'{self.save_dir}/global_plot_feat_{self.index+1}.html', embed_options={'renderer': 'svg'})

            self.index += 1


    def get_background(self, X, num_feats=[], cat_feats=[]):
        if self.background == "median":
            self.background = "event"
        return getattr(utils, "calc_avg_" + self.background)(X, numerical_feats=num_feats, categorical_feats=cat_feats)

