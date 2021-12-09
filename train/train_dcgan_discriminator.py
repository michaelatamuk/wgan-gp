import numpy as np

from torch.autograd import Variable
import torch

from train.compute_gradient_penalty import compute_gradient_penalty
from train.params import Params
from utils import get_tensors_type


def train_discriminator(params: Params, real_images, generated_images, valid, fake):
    # Configure input
    real_images_as_tensor = Variable(real_images.type(get_tensors_type()))

    params.discriminator_optimizer.zero_grad()

    # Measure discriminator's ability to classify real from generated samples
    result_real = params.discriminator(real_images_as_tensor)
    result_fake = params.discriminator(generated_images.detach())
    real_loss = params.loss_function(result_real, valid)
    fake_loss = params.loss_function(result_fake, fake)
    loss = (real_loss + fake_loss) / 2

    loss.backward()
    params.discriminator_optimizer.step()
    return loss
