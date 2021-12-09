import argparse

from models.type import Type as ModelType
from data.type import Type as DataType
from train.params import Params
from train.builder import build_params

import ssl

from train.train_wgan import TrainWGan

ssl._create_default_https_context = ssl._create_unverified_context

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--gradient_penalty_lambda", type=int, default=10, help="loss weight for gradient penalty")
parser.add_argument("--save_generated_image_every", type=int, default=50, help="interval batches between saving image")
args = parser.parse_args()
print(args)

params: Params = build_params(args, ModelType.WGAN_GP, DataType.FASHION_MNIST)

train: TrainWGan = TrainWGan(params)
train.run()

