import numpy as np
from sklearn.covariance import EmpiricalCovariance

class Mahalanobis():
    def __init__(self):
        super(Mahalanobis, self).__init__()

        self.name = "Mahalanobis"
        self.classes = None
        self.of = None

    def clear(self):
        self.classes = None
        self.of = None

    def fit(self, df):
        self.classes = list(df["original_label"].unique())
        self.of = {}

        for index in self.classes:
            x = df[df["original_label"] == index]["features"].tolist()
            self.of[index] = EmpiricalCovariance().fit(x)

    def test(self, df):
        x = df["features"].tolist()
        res = np.stack([np.sqrt(self.of[index].mahalanobis(x)) for index in self.classes], axis=0)
        res = res.min(axis=0)
        return -res

    def verify(self, out, p):
        return out <= p     
