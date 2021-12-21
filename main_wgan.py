import torch.nn
from torch import Tensor
from torchvision.utils import save_image

from models.type import Type as ModelType
from data.type import Type as DataType
from results.results import Results
from train.params import Params
from train.builder import build_params

import ssl

from train.train_wgan import TrainWGan

ssl._create_default_https_context = ssl._create_unverified_context

args: {} = {}
args["epochs"] = 200 # number of epochs of training
args["batch_size"] = 64 # size of the batches
args["lr"] = 0.0002 # adam: learning rate
args["b1"] = 0.5 # adam: decay of first order momentum of gradient
args["b2"] = 0.999 # adam: decay of first order momentum of gradient
args["latent_dim"] = 100 # dimensionality of the latent space
args["critic"] = 5 # number of training steps for discriminator per iter
args["gradient_penalty_lambda"] = 10 # loss weight for gradient penalty
args["save_generated_image_every"] = 50 # interval batches between saving image

params: Params = build_params(args, ModelType.WGAN_GP, DataType.FASHION_MNIST)

results: Results = Results("wgan")

train: TrainWGan = TrainWGan(params, results.loss_updated_callback)
train.run()

generator_losses: {} = results.generator_losses
discriminator_losses: {} = results.discriminator_losses
last_generator: torch.nn.Module = results.last_generator

fixed_noise: Tensor = torch.randn(2, 100)
if torch.cuda.is_available():
    fixed_noise = fixed_noise.cuda()

generated_image = last_generator(fixed_noise)
image_path_1 = "images/result_wgan_1.png"
image_path_2 = "images/result_wgan_2.png"
save_image(generated_image.data[0], image_path_1, nrow=2, normalize=True)
save_image(generated_image.data[1], image_path_2, nrow=2, normalize=True)
