from abc import ABC, abstractmethod

import torch

from train.params import Params


class TrainBase(ABC):
    def __init__(self, params: Params):
        self.params = params
        self.batches_done = 0

    def run(self):
        for epoch in range(self.params.epochs):
            for batch_index, (real_images, _) in enumerate(self.params.dataloader):
                self.train_step(epoch, batch_index, real_images)

    def train_begin(self):
        pass

    @abstractmethod
    def train_step(self, epoch, batch_index, real_images):
        pass

    @staticmethod
    def get_is_cuda():
        return True if torch.cuda.is_available() else False

    @staticmethod
    def get_tensors_type():
        return torch.cuda.FloatTensor if TrainBase.get_is_cuda() else torch.FloatTensor
