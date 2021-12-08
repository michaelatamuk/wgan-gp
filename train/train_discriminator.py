import numpy as np

from torch.autograd import Variable
import torch

from train.compute_gradient_penalty import compute_gradient_penalty
from utils import get_tensors_type


def train_discriminator(images, params):
    params.discriminator_optimizer.zero_grad()

    # Sample noise as generator input
    noise = Variable(get_tensors_type()(np.random.normal(0, 1, (images.shape[0], params.latent_dim))))

    # Generate a batch of images
    fake_images = params.generator(noise)

    # Real images
    tensor_type = get_tensors_type()
    images_type = images.type(tensor_type)
    real_images = Variable(images_type)
    real_validity = params.discriminator(real_images)

    # Fake images
    fake_validity = params.discriminator(fake_images)

    # Gradient penalty
    gradient_penalty = compute_gradient_penalty(params.discriminator, real_images.data, fake_images.data)

    # Adversarial loss
    loss = -torch.mean(real_validity) + torch.mean(fake_validity) + params.gradient_penalty_lambda * gradient_penalty
    loss.backward()
    params.discriminator_optimizer.step()
    params.generator_optimizer.zero_grad()
    return loss, noise
