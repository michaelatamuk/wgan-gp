import os
from argparse import Namespace

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets
import torch
from torchvision.transforms import Normalize, Resize, ToTensor, Compose

from models.type import Type as ModelType
from data.type import Type as DataType

from models.wgan.discriminator import Discriminator as Discriminator_WGAN
from models.wgan.generator import Generator as Generator_WGAN
from models.dcgan.discriminator import Discriminator as Discriminator_DCGAN
from models.dcgan.generator import Generator as Generator_DCGAN
from train.params import Params


def build_params(args: {}, model_type: ModelType, data_type: DataType):
    # Configure data loader
    transforms = Compose([Resize(32), ToTensor(), Normalize([0.5], [0.5])])
    if data_type == DataType.CIFAR10:
        os.makedirs("data/cifar10", exist_ok=True)
        dataset = datasets.CIFAR10("data/cifar10", train=True, download=True, transform=transforms)
    elif data_type == DataType.MNIST:
        os.makedirs("data/mnist", exist_ok=True)
        dataset = datasets.MNIST("data/mnist", train=True, download=True, transform=transforms)
    else:
        os.makedirs("data/fashion_mnist", exist_ok=True)
        dataset = datasets.FashionMNIST("data/fashion_mnist", train=True, download=True, transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args["batch_size"], shuffle=True)

    dataiter = iter(dataloader)
    images, _ = dataiter.next()
    images_channels: int = images[0].size()[0]
    images_width: int = images[0].size()[1]
    images_height: int = images[0].size()[2]

    os.makedirs("images", exist_ok=True)

    # Create generator and discriminator
    generator: torch.nn.Module = None
    discriminator: torch.nn.Module = None
    if model_type == ModelType.DCGAN:
        generator = Generator_DCGAN(args["latent_dim"], images_channels, images_width, images_height)
        discriminator = Discriminator_DCGAN(images_channels, images_width, images_height)
    else:
        generator = Generator_WGAN(args["latent_dim"], images_channels, images_width, images_height)
        discriminator = Discriminator_WGAN(images_channels, images_width, images_height)

    # Create Loss Function
    loss_function: torch.nn.Module = None
    if model_type == ModelType.DCGAN:
        loss_function = torch.nn.BCELoss()

    # Create Optimizers
    generator_optimizer: Optimizer = torch.optim.Adam(generator.parameters(), lr=args["lr"],
                                                      betas=(args["b1"], args["b2"]))
    discriminator_optimizer: Optimizer = torch.optim.Adam(discriminator.parameters(), lr=args["lr"],
                                                          betas=(args["b1"], args["b2"]))

    # Fill Train Params
    params: Params = Params()
    params.epochs = args["epochs"]
    params.dataloader = dataloader
    params.generator = generator
    params.discriminator = discriminator
    params.generator_optimizer = generator_optimizer
    params.discriminator_optimizer = discriminator_optimizer
    if "gradient_penalty_lambda" in args:
        params.gradient_penalty_lambda = args["gradient_penalty_lambda"]
    params.loss_function = loss_function
    params.latent_dim = args["latent_dim"]
    if "critic" in args:
        params.critic = args["critic"]
    params.save_generated_image_every = args["save_generated_image_every"]

    return params
