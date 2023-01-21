import numpy as np
import torch
from .strategy import Strategy

class EntropySampling(Strategy):
    def __init__(self, dataset, net):
        super(EntropySampling, self).__init__(dataset, net)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)
        probs_1 = 1-probs 
        log_probs = torch.log(probs)
        log_probs_1 = torch.log(probs_1)
        uncertainties = (probs*log_probs)
        uncertainties_1 = (probs_1*log_probs_1)
        uncertainties = uncertainties + uncertainties_1
        uncertainties = uncertainties.sum([1,2,3])
        return unlabeled_idxs[uncertainties.sort()[1][:n]]