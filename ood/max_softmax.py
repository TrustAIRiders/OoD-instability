import numpy as np
from sklearn.utils.extmath import softmax

class MaxSoftmax():
    def __init__(self):
        super(MaxSoftmax, self).__init__()

        self.name = "MaxSoftmax"

    def clear(self):
        pass

    def fit(self, df):
        pass

    def verify(self, out, p):
        return out <= p   

    def test(self, df):
        classifier = np.array(df["classifier"].tolist())
        return np.asarray(np.max(softmax(np.asarray(classifier)), axis=1))
