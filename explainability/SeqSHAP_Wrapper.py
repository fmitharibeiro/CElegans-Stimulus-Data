import numpy as np
from .seqshap.SeqShapKernel import SeqShapKernel



class SeqSHAP_Explainer:
    def __init__(self, model=None, dataset:str="", seed=None, **kwargs):
        self.dataset = dataset
        self.model = model
        self.index = 0
        self.seed = seed
        self.save_dir = f"plots/{self.dataset}/SeqSHAP"
        self.f = lambda x: self.model.predict(x)[:, :, self.index]

    def __call__(self, X, *args, **kwargs):
        while self.index < X.shape[2]:
            for i in range(X.shape[0]):
                # TODO: try changing the background!
                kernel = SeqShapKernel(self.f, X, i, self.index, self.dataset, background="feat_mean", random_seed=self.seed)

                kernel(X[i])

            self.index += 1
