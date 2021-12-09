import os
from argparse import Namespace

from torch.utils.data import DataLoader
from torchvision import datasets
import torch
from torchvision.transforms import Normalize, Resize, ToTensor, Compose

from models.type import Type
from models.wgan_discriminator import Discriminator as Discriminator_WGAN
from models.wgan_generator import Generator as Generator_WGAN
from models.dcgan_discriminator import Discriminator as Discriminator_DCGAN
from models.dcgan_generator import Generator as Generator_DCGAN
from train.weight_init_normal import weights_init_normal
from train.params import Params
from utils import get_is_cuda


def build_params(args: Namespace, network_type: Type):
    images_shape = (args.channels, args.img_size, args.img_size)

    # Loss weight for gradient penalty
    gradient_penalty_lambda = 10

    # Initialize generator and discriminator
    generator: torch.nn.Module = None
    discriminator: torch.nn.Module = None
    if network_type == Type.DCGAN:
        generator = Generator_DCGAN(args.latent_dim, args.img_size, args.channels)
        discriminator = Discriminator_DCGAN(args.img_size, args.channels)

        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    else:
        generator = Generator_WGAN(args.latent_dim, images_shape)
        discriminator = Discriminator_WGAN(images_shape)

    loss_function: torch.nn.Module = None
    if network_type == Type.DCGAN:
        loss_function = torch.nn.BCELoss()

    if get_is_cuda():
        generator.cuda()
        discriminator.cuda()
        if loss_function is not None:
            loss_function.cuda()

    # Configure data loader
    transforms = Compose([Resize(args.img_size), ToTensor(), Normalize([0.5], [0.5])])

    if args.data == "cifar10":
        os.makedirs("data/cifar10", exist_ok=True)
        dataset = datasets.CIFAR10("data/cifar10", train=True, download=True, transform=transforms)
    else:
        os.makedirs("data/mnist", exist_ok=True)
        dataset = datasets.MNIST("data/mnist", train=True, download=True, transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    os.makedirs("images", exist_ok=True)

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
    params.loss_function = loss_function
    params.latent_dim = args.latent_dim
    if "n_critic" in args:
        params.critic = args.n_critic
    params.sample_interval = args.sample_interval

    return params
