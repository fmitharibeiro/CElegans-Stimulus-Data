from timeshap.wrappers import TimeSHAPWrapper



class TimeShap_Wrapper(TimeSHAPWrapper):
    def __init__(self, seed, model=None, batch_budget=0):
        super().__init__(model, batch_budget)

        self.seed = seed
        self.param_grid = {}

    def fit(self, X, y):
        pass