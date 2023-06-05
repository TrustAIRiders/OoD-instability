import numpy as np
import faiss

class KNN():
    def __init__(self):
        super(KNN, self).__init__()

        self.name = "KNN"
        self.index = None
        self.K = 50

    def clear(self):
        self.index = None

    def fit(self, df):
        features = np.array(df["features"].tolist())
        self.index = faiss.IndexFlatL2(features.shape[1])
        self.index.add(features)

    def test(self, df):
        features = np.array(df["features"].tolist())
        
        D, _ = self.index.search(features, self.K)
        scores = -D[:,-1]
        return scores

    def verify(self, out, p):
        return out <= p     
