import numpy as np

class MaxLogits():
    def __init__(self):
        super(MaxLogits, self).__init__()

        self.name = "MaxLogits"
        self.max = None

    def clear(self):
        self.max = None

    def fit(self, df):
        classifier = np.array(df["classifier"].tolist())
        self.max = np.max(np.max(classifier))

    def verify(self, out, p):
        return out <= self.max * p

    def test(self, df):
        classifier = np.array(df["classifier"].tolist())
        return np.max(classifier, axis=1)
