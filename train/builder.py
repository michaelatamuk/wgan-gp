import os
from argparse import Namespace

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets
import torch
from torchvision.transforms import Normalize, Resize, ToTensor, Compose

from models.type import Type
from models.wgan.discriminator import Discriminator as Discriminator_WGAN
from models.wgan.generator import Generator as Generator_WGAN
from models.dcgan.discriminator import Discriminator as Discriminator_DCGAN
from models.dcgan.generator import Generator as Generator_DCGAN
from train.params import Params


def build_params(args: Namespace, network_type: Type):
    images_shape = (args.channels, args.img_size, args.img_size)

    # Create generator and discriminator
    generator: torch.nn.Module = None
    discriminator: torch.nn.Module = None
    if network_type == Type.DCGAN:
        generator = Generator_DCGAN(args.latent_dim, args.img_size, args.channels)
        discriminator = Discriminator_DCGAN(args.img_size, args.channels)
    else:
        generator = Generator_WGAN(args.latent_dim, images_shape)
        discriminator = Discriminator_WGAN(images_shape)

    # Create Loss Function
    loss_function: torch.nn.Module = None
    if network_type == Type.DCGAN:
        loss_function = torch.nn.BCELoss()

    # Configure data loader
    transforms = Compose([Resize(args.img_size), ToTensor(), Normalize([0.5], [0.5])])

    if args.data == "cifar10":
        os.makedirs("data/cifar10", exist_ok=True)
        dataset = datasets.CIFAR10("data/cifar10", train=True, download=True, transform=transforms)
    elif args.data == "cifar10":
        os.makedirs("data/mnist", exist_ok=True)
        dataset = datasets.MNIST("data/mnist", train=True, download=True, transform=transforms)
    else:
        os.makedirs("data/fashion_mnist", exist_ok=True)
        dataset = datasets.FashionMNIST("data/fashion_mnist", train=True, download=True, transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    os.makedirs("images", exist_ok=True)

    # Create Optimizers
    generator_optimizer: Optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    discriminator_optimizer: Optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # Fill Train Params
    params: Params = Params()
    params.epochs = args.n_epochs
    params.dataloader = dataloader
    params.generator = generator
    params.discriminator = discriminator
    params.generator_optimizer = generator_optimizer
    params.discriminator_optimizer = discriminator_optimizer
    if "gradient_penalty_lambda" in args:
        params.gradient_penalty_lambda = args.gradient_penalty_lambda
    params.loss_function = loss_function
    params.latent_dim = args.latent_dim
    if "n_critic" in args:
        params.critic = args.n_critic
    params.save_generated_image_every = args.save_generated_image_every

    return params
