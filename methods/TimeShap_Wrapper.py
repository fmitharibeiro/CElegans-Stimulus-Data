from .TimeSHAP.KerasModelWrapper import KerasModelWrapper



class TimeShap_Wrapper(KerasModelWrapper):
    def __init__(self, seed, model=None, batch_budget=750000):
        super().__init__(model, batch_budget)

        self.seed = seed

    def fit(self, X, y):
        # TODO: Check if hidden layers are working
        f_hs = lambda x, y=None: super().predict_last_hs(x, y)

        pass

    def set_params(self, **params):
        if not params:
            return self
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        return self

    def get_params(self):
        return {attr: getattr(self, attr)
                for attr in dir(self)
                if not callable(getattr(self, attr)) and not attr.startswith("__")}