from enum import Enum

import torch.nn as nn


class Type(Enum):
    CIFAR10 = 1
    MNIST = 2
    FASHION_MNIST = 3
