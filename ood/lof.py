import numpy as np
from sklearn.neighbors import LocalOutlierFactor as LOFScikit

class LOF():
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"

    def __init__(self, distance):
        super(LOF, self).__init__()

        self.distance = distance
        self.name = "lof_{}".format(distance)

        self.of = None

    def clear(self):
        self.of = None

    def fit(self, df):
        features = df["features"].tolist()
        self.of = LOFScikit(contamination=1e-12, novelty=True, metric=self.distance, n_jobs=-1)
        self.of.fit(features)

    def test(self, df):
        features = df["features"].tolist()
        return self.of.score_samples(features)

    def verify(self, out, p):
        return out <= np.percentile(self.of.negative_outlier_factor_, 100. * p)
