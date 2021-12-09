import numpy as np
import torch
from torch.autograd import Variable

from train.params import Params
from train.train_base import TrainBase


class TrainDCGan(TrainBase):
    def __init__(self, params: Params):
        super(TrainDCGan, self).__init__(params)

    def get_train_name(self):
        return "dcgan"

    def train_begin(self):
        super(TrainDCGan, self).train_begin()

        def weights_init_normal(m):
            classname = m.__class__.__name__

            if classname.find("Conv") != -1:
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

            elif classname.find("BatchNorm2d") != -1:
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

        # Initialize weights
        self.params.generator.apply(weights_init_normal)
        self.params.discriminator.apply(weights_init_normal)

    def train_step(self, epoch, batch_index, real_images):
        # Ground truths
        valid = Variable(self.get_tensors_type()(real_images.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(self.get_tensors_type()(real_images.shape[0], 1).fill_(0.0), requires_grad=False)

        self.batches_done = epoch * len(self.params.dataloader) + batch_index
        generator_loss, generated_images = self.train_generator(real_images, valid)

        discriminator_loss = self.train_discriminator(real_images, generated_images, valid, fake)

        self.print_results(epoch, batch_index, discriminator_loss, generator_loss)

    def train_generator(self, images, valid):
        self.params.generator_optimizer.zero_grad()

        # Sample noise as generator input
        noise = Variable(self.get_tensors_type()(np.random.normal(0, 1, (images.shape[0], self.params.latent_dim))))

        # Generate a batch of images
        generated_images = self.params.generator(noise)

        # Loss measures generator's ability to fool the discriminator
        generated_result = self.params.discriminator(generated_images)
        loss = self.params.loss_function(generated_result, valid)

        loss.backward()
        self.params.generator_optimizer.step()

        self.save_generated_images(generated_images)

        return loss, generated_images

    def train_discriminator(self, real_images, generated_images, valid, fake):
        # Configure input
        real_images_as_tensor = Variable(real_images.type(self.get_tensors_type()))

        self.params.discriminator_optimizer.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_result = self.params.discriminator(real_images_as_tensor)
        real_loss = self.params.loss_function(real_result, valid)

        generated_result = self.params.discriminator(generated_images.detach())
        generated_loss = self.params.loss_function(generated_result, fake)

        loss = (real_loss + generated_loss) / 2
        loss.backward()

        self.params.discriminator_optimizer.step()
        return loss
