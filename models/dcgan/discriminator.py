import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, images_channels: int, images_width: int, images_height: int):
        super(Discriminator, self).__init__()

        def discriminator_block(in_channels: int, out_channels: int, bn: bool = True):
            block = [nn.Conv2d(in_channels, out_channels, 3, 2, 1),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_channels, 0.8))
            return block

        self.model = nn.Sequential(*discriminator_block(images_channels, 16, bn=False),
                                   *discriminator_block(16, 32),
                                   *discriminator_block(32, 64),
                                   *discriminator_block(64, 128))

        # The height and width of downsampled image
        ds_size = images_width // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1),
                                       nn.Sigmoid())

    def forward(self, image):
        out = self.model(image)
        out = out.view(out.shape[0], -1)
        result = self.adv_layer(out)
        return result
