import numpy as np
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, images_channels: int, images_width: int, images_height: int):
        super(Generator, self).__init__()
        self.images_shape = (images_channels, images_width, images_height)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(*block(latent_dim, 128, normalize=False),
                                   *block(128, 256),
                                   *block(256, 512),
                                   *block(512, 1024),
                                   nn.Linear(1024, int(np.prod(self.images_shape))),
                                   nn.Tanh())

    def forward(self, noise):
        image = self.model(noise)
        image = image.view(image.shape[0], *self.images_shape)
        return image
