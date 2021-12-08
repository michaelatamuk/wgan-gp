import argparse
import os

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
import torch

from models.discriminator import Discriminator
from models.generator import Generator
from train.params import Params
from train.train import train
from utils import get_is_cuda

os.makedirs("images", exist_ok=True)

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
args = parser.parse_args()
print(args)

img_shape = (args.channels, args.img_size, args.img_size)


# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator(args.latent_dim, img_shape)
discriminator = Discriminator(img_shape)

if get_is_cuda():
    generator.cuda()
    discriminator.cuda()

# Configure data loader
transforms = transforms.Compose([transforms.Resize(args.img_size),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5], [0.5])])

dataset = datasets.CIFAR10("data/cifar10", train=True, download=True, transform=transforms)

os.makedirs("../../data/cifar10", exist_ok=True)
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
params.gradient_penalty_lambda = lambda_gp
params.latent_dim = args.latent_dim
params.critic = args.n_critic
params.sample_interval = args.sample_interval

train(params)
