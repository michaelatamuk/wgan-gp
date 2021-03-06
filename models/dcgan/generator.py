import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, images_channels: int, images_width: int, images_height: int):
        super(Generator, self).__init__()

        self.init_size = images_width // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(nn.BatchNorm2d(128),
                                         nn.Upsample(scale_factor=2),
                                         nn.Conv2d(128, 128, 3, stride=1, padding=1),
                                         nn.BatchNorm2d(128, 0.8),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Upsample(scale_factor=2),
                                         nn.Conv2d(128, 64, 3, stride=1, padding=1),
                                         nn.BatchNorm2d(64, 0.8),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Conv2d(64, images_channels, 3, stride=1, padding=1),
                                         nn.Tanh())

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        image = self.conv_blocks(out)
        return image
