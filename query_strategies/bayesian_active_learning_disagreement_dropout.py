import numpy as np
import torch
from .strategy import Strategy

class BALDDropout(Strategy):
    def __init__(self, dataset, net, n_drop=10):
        super(BALDDropout, self).__init__(dataset, net)
        self.n_drop = n_drop

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob_dropout_split(unlabeled_data, n_drop=self.n_drop)
        return unlabeled_idxs[probs.sort(descending=True)[:n]]
