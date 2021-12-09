from enum import Enum

import torch.nn as nn


class Type(Enum):
    DCGAN = 1
    WGAN_GP = 2
