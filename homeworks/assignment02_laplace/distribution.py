import numpy as np

class LaplaceDistribution:    
    @staticmethod
    def mean_abs_deviation_from_median(x: np.ndarray):
        mean = np.median(x, axis=0)
        dev = np.sum(np.abs(x - mean), axis=0) / x.shape[0]

        return dev

    def __init__(self, features):
        self.loc = np.median(features, axis=0)
        self.scale = self.mean_abs_deviation_from_median(features)


    def logpdf(self, values):
        return np.log(1/2/self.scale) - np.abs(values - self.loc)/self.scale
        
    
    def pdf(self, values):
        return np.exp(self.logpdf(values))
