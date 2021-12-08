import numpy as np

from torch.autograd import Variable
import torch

from train.compute_gradient_penalty import compute_gradient_penalty
from utils import get_tensors_type


def train_discriminator(imgs, params):
    params.discriminator_optimizer.zero_grad()

    # Sample noise as generator input
    noise = Variable(get_tensors_type()(np.random.normal(0, 1, (imgs.shape[0], params.latent_dim))))

    # Generate a batch of images
    fake_imgs = params.generator(noise)

    # Real images
    tensor_type = get_tensors_type()
    imgs_type = imgs.type(tensor_type)
    real_imgs = Variable(imgs_type)
    real_validity = params.discriminator(real_imgs)

    # Fake images
    fake_validity = params.discriminator(fake_imgs)

    # Gradient penalty
    gradient_penalty = compute_gradient_penalty(params.discriminator, real_imgs.data, fake_imgs.data)

    # Adversarial loss
    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + params.gradient_penalty_lambda * gradient_penalty
    d_loss.backward()
    params.discriminator_optimizer.step()
    params.generator_optimizer.zero_grad()
    return d_loss, noise
