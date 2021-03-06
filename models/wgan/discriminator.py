import torch.nn as nn
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, images_channels: int, images_width: int, images_height: int):
        super(Discriminator, self).__init__()

        images_shape = (images_channels, images_width, images_height)
        self.model = nn.Sequential(nn.Linear(int(np.prod(images_shape)), 512),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Linear(512, 256),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Linear(256, 1))

    def forward(self, image):
        image_flat = image.view(image.shape[0], -1)
        result = self.model(image_flat)
        return result
