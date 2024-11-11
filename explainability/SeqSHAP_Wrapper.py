from .seqshap.SeqShapKernel import SeqShapKernel
from .timeshap.wrappers import TorchModelWrapper, TensorFlowModelWrapper


class SeqSHAP_Explainer:
    def __init__(self, model=None, dataset:str="", use_hidden:bool=False, seed=None, **kwargs):
        self.dataset = dataset
        self.model = model
        self.index = 0
        self.seed = seed
        self.save_dir = f"plots/{self.dataset}/SeqSHAP"
        self.segmentation = getattr(kwargs.get('other_args'), 'segmentation')

        self.nsamples = 2**15
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
            self.f = lambda x: self.model.predict(x)[:, :, self.index]

    def __call__(self, X, *args, **kwargs):
        while self.index < X.shape[2]:
            for i in range(X.shape[0]):
                kernel = SeqShapKernel(self.f, X, i, self.index, self.dataset, background="median", random_seed=self.seed, nsamples=self.nsamples, segmentation=self.segmentation)

                kernel(X[i])

            self.index += 1
