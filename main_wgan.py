import argparse

from models.type import Type
from train.params import Params
from train.builder import build_params

import ssl

from train.train_wgan import TrainWGan

ssl._create_default_https_context = ssl._create_unverified_context

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--data", type=str, default='cifar10', help="data to train")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--gradient_penalty_lambda", type=int, default=10, help="loss weight for gradient penalty")
parser.add_argument("--sample_interval", type=int, default=50, help="interval betwen image samples")
args = parser.parse_args()
print(args)

params: Params = build_params(args, Type.WGAN_GP)

train: TrainWGan = TrainWGan(params)
train.run()

