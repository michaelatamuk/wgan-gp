import os
from argparse import Namespace

from torch.utils.data import DataLoader
from torchvision import datasets
import torch
from torchvision.transforms import Normalize, Resize, ToTensor, Compose

from models.discriminator import Discriminator
from models.generator import Generator
from train.params import Params
from utils import get_is_cuda


def build_params(args: Namespace):
    images_shape = (args.channels, args.img_size, args.img_size)

    # Loss weight for gradient penalty
    gradient_penalty_lambda = 10

    # Initialize generator and discriminator
    generator = Generator(args.latent_dim, images_shape)
    discriminator = Discriminator(images_shape)

    if get_is_cuda():
        generator.cuda()
        discriminator.cuda()

    # Configure data loader
    transforms = Compose([Resize(args.img_size), ToTensor(), Normalize([0.5], [0.5])])

    os.makedirs("data/cifar10", exist_ok=True)
    dataset = datasets.CIFAR10("data/cifar10", train=True, download=True, transform=transforms)

    os.makedirs("images", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    params: Params = Params()
    params.epochs = args.n_epochs
    params.dataloader = dataloader
    params.generator = generator
    params.discriminator = discriminator
    params.generator_optimizer = optimizer_G
    params.discriminator_optimizer = optimizer_D
    params.gradient_penalty_lambda = gradient_penalty_lambda
    params.latent_dim = args.latent_dim
    params.critic = args.n_critic
    params.sample_interval = args.sample_interval

    return params
