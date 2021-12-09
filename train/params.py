import torch.nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class Params:
    def __init__(self):
        self.epochs: int = 0
        self.dataloader: DataLoader = None
        self.generator: torch.nn.Module = None
        self.discriminator: torch.nn.Module = None
        self.generator_optimizer: Optimizer = None
        self.discriminator_optimizer: Optimizer = None
        self.gradient_penalty_lambda: int = None
        self.loss_function: torch.nn.Module = None
        self.latent_dim: int = None
        self.critic: int = None
        self.save_generated_image_every: int = None
