import numpy as np
import torch
from torch.autograd import Variable
from torchvision.utils import save_image

from train.params import Params
from utils import get_tensors_type


def train_generator(batches_done, images, valid, params: Params):
    params.generator_optimizer.zero_grad()

    # Sample noise as generator input
    noise = Variable(get_tensors_type()(np.random.normal(0, 1, (images.shape[0], params.latent_dim))))

    # Generate a batch of images
    generated_images = params.generator(noise)

    # Loss measures generator's ability to fool the discriminator
    loss = params.loss_function(params.discriminator(generated_images), valid)

    loss.backward()
    params.generator_optimizer.step()

    if batches_done % params.sample_interval == 0:
        save_image(generated_images.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

    return loss, generated_images
