import numpy as np

class FreeEnergy():
    def __init__(self, T=1.0):
        super(FreeEnergy, self).__init__()

        self.name = "FreeEnergy_t={}".format(T)
        self.T = T

    def clear(self):
        pass

    def fit(self, df):
        pass

    def verify(self, out, p):
        return out <= p   

    def test(self, df):
        logits = np.array(df["classifier"].tolist())
        return np.log(np.sum(np.exp(logits), axis=1))
