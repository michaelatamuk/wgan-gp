import numpy as np
import torch
from torch import autograd
from torch.autograd import Variable

from train.params import Params
from train.train_base import TrainBase


class TrainWGan(TrainBase):
    def __init__(self, params: Params):
        super(TrainWGan, self).__init__(params)

    def get_train_name(self):
        return "wgan"

    def train_step(self, epoch, batch_index, real_images):
        discriminator_loss, noise = self.train_discriminator(real_images)

        # Train the generator every critic steps
        if batch_index % self.params.critic == 0:
            generator_loss = self.train_generator(noise)
            self.batches_done += self.params.critic

            self.print_results(epoch, batch_index, discriminator_loss, generator_loss)

    def train_generator(self, noise):
        # Generate a batch of images
        generated_images = self.params.generator(noise)

        # Train on generated images
        generated_result = self.params.discriminator(generated_images)

        # Loss measures generator's ability to fool the discriminator
        loss = -torch.mean(generated_result)
        loss.backward()

        self.params.generator_optimizer.step()

        self.save_generated_image(generated_images)

        return loss

    def train_discriminator(self, real_images):
        self.params.discriminator_optimizer.zero_grad()

        # Sample noise as generator input
        noise = Variable(
            self.get_tensors_type()(np.random.normal(0, 1, (real_images.shape[0], self.params.latent_dim))))

        # Generate a batch of images
        generated_images = self.params.generator(noise)

        # Real images
        tensor_type = self.get_tensors_type()
        real_images_type = real_images.type(tensor_type)
        real_images_as_tensor = Variable(real_images_type)
        real_result = self.params.discriminator(real_images_as_tensor)

        generated_result = self.params.discriminator(generated_images)

        # Gradient penalty
        gradient_penalty = self.compute_gradient_penalty(self.params.discriminator, real_images.data,
                                                         generated_images.data)

        loss = -torch.mean(real_result) + torch.mean(generated_result) + \
               self.params.gradient_penalty_lambda * gradient_penalty
        loss.backward()

        self.params.discriminator_optimizer.step()

        self.params.generator_optimizer.zero_grad()
        return loss, noise

    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        """Calculate the gradient penalty loss for WGAN GP"""

        # Random weight term for interpolation between real and fake samples
        alpha = self.get_tensors_type()(np.random.random((real_samples.size(0), 1, 1, 1)))

        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = Variable(self.get_tensors_type()(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)

        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(outputs=d_interpolates,
                                  inputs=interpolates,
                                  grad_outputs=fake,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
