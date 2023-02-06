import numpy as np
import torch
from .strategy import Strategy

class IdealActiveLearning(Strategy):
    def __init__(self, dataset, net):
        super(IdealActiveLearning, self).__init__(dataset, net)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        losses_by_imgs = self.get_loss_by_img(unlabeled_data)
        return unlabeled_idxs[losses_by_imgs.sort()[1][:n]] # 

