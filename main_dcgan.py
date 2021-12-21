from models.type import Type as ModelType
from data.type import Type as DataType
from results.results import Results
from train.params import Params
from train.builder import build_params

import ssl

from train.train_dcgan import TrainDCGan

ssl._create_default_https_context = ssl._create_unverified_context

args: {} = {}
args["epochs"] = 200 # number of epochs of training
args["batch_size"] = 64 # size of the batches
args["lr"] = 0.0002 # adam: learning rate
args["b1"] = 0.5 # adam: decay of first order momentum of gradient
args["b2"] = 0.999 # adam: decay of first order momentum of gradient
args["latent_dim"] = 100 # dimensionality of the latent space
args["save_generated_image_every"] = 50 # interval batches between saving image

params: Params = build_params(args, ModelType.DCGAN, DataType.MNIST)

results: Results = Results("dcgan")

train = TrainDCGan(params, results.loss_updated_callback)
train.run()
