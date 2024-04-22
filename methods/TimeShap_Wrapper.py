from timeshap.explainer.kernel import TimeShapKernel



class TimeShap_Wrapper(TimeShapKernel):
    def __init__(self, model, background, seed, mode, varying=None, link="identity", **kwargs):
        super().__init__(model, background, seed, mode, varying, link, **kwargs)
    
    def f(self, X):
        """
        X: (# samples x # sequence length x # features)
        returns: output of the model for those samples, (# samples x # model outputs).
        In order to use TimeSHAP in an optimized way, this model can also return the explained
        model's hidden state.
        """