import numpy as np
import torch
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import save_image

from train.params import Params
from train.train_base import TrainBase


class TrainWGan(TrainBase):
    def __init__(self, params: Params):
        super(TrainWGan, self).__init__(params)

    def train_step(self, epoch, batch_index, real_images):
        discriminator_loss, noise = self.train_discriminator(real_images)

        # Train the generator every critic steps
        if batch_index % self.params.critic == 0:
            generator_loss = self.train_generator(noise)
            self.batches_done += self.params.critic

            print("[Epoch %d/%d] [Batch %d/%d] [Discriminator loss: %f] [Generator loss: %f]"
                  % (epoch + 1, self.params.epochs, batch_index, len(self.params.dataloader),
                     discriminator_loss.item(), generator_loss.item()))

    def train_generator(self, noise):
        # Generate a batch of images
        fake_images = self.params.generator(noise)

        # Loss measures generator's ability to fool the discriminator
        # Train on fake images
        fake_validity = self.params.discriminator(fake_images)
        loss = -torch.mean(fake_validity)
        loss.backward()
        self.params.generator_optimizer.step()

        if self.batches_done % self.params.sample_interval == 0:
            save_image(fake_images.data[:25], "images/%d.png" % self.batches_done, nrow=5, normalize=True)

        return loss

    def train_discriminator(self, real_images):
        self.params.discriminator_optimizer.zero_grad()

        # Sample noise as generator input
        noise = Variable(self.get_tensors_type()(np.random.normal(0, 1, (real_images.shape[0], self.params.latent_dim))))

        # Generate a batch of images
        fake_images = self.params.generator(noise)

        # Real images
        tensor_type = self.get_tensors_type()
        real_images_type = real_images.type(tensor_type)
        real_images_as_tensor = Variable(real_images_type)
        real_validity = self.params.discriminator(real_images_as_tensor)

        # Fake images
        fake_validity = self.params.discriminator(fake_images)

        # Gradient penalty
        gradient_penalty = self.compute_gradient_penalty(self.params.discriminator, real_images.data, fake_images.data)

        # Adversarial loss
        loss = -torch.mean(real_validity) + torch.mean(
            fake_validity) + self.params.gradient_penalty_lambda * gradient_penalty
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
