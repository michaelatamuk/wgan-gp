from abc import ABC, abstractmethod
from collections import Callable

import torch
from torch import Tensor
from torchvision.utils import save_image

from train.params import Params


class TrainBase(ABC):
    def __init__(self, params: Params, loss_updated_callback: Callable = None):
        self.params = params
        self.batches_done = 0
        self.loss_updated_callback: Callable = loss_updated_callback

    def run(self):
        self.train_begin()
        for epoch in range(self.params.epochs):
            for batch_index, (real_images, _) in enumerate(self.params.dataloader):
                self.train_step(epoch, batch_index, real_images)

    def train_begin(self):
        if self.get_is_cuda():
            self.params.generator.cuda()
            self.params.discriminator.cuda()
            if self.params.loss_function is not None:
                self.params.loss_function.cuda()

    @abstractmethod
    def train_step(self, epoch, batch_index, real_images):
        pass

    def save_generated_images(self, generated_images):
        if self.batches_done % self.params.save_generated_image_every == 0:
            image_path = "images/" + self.get_train_name() + "_%d.png" % self.batches_done
            save_image(generated_images.data, image_path, nrow=8, normalize=True)

    @abstractmethod
    def get_train_name(self):
        pass

    def call_loss_updated_callback(self, epoch: int, batch_index: int,
                                   discriminator_loss: Tensor, generator_loss: Tensor):
        step_num = epoch * len(self.params.dataloader) + (batch_index + 1)
        if self.loss_updated_callback is not None:
            self.loss_updated_callback(step_num, self.params.generator,
                                       generator_loss.item(), discriminator_loss.item())

    def print_results(self, epoch: int, batch_index: int,
                                   discriminator_loss: Tensor, generator_loss: Tensor):
        print("[Epoch %d/%d] [Batch %d/%d] [Discriminator loss: %f] [Generator loss: %f]"
              % (epoch + 1, self.params.epochs, batch_index+1, len(self.params.dataloader),
                 discriminator_loss.item(), generator_loss.item()))

    @staticmethod
    def get_is_cuda():
        return True if torch.cuda.is_available() else False

    @staticmethod
    def get_tensors_type():
        return torch.cuda.FloatTensor if TrainBase.get_is_cuda() else torch.FloatTensor
