import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image

from utils import get_tensors_type

from train.params import Params
from train.train_base import TrainBase


class TrainDCGan(TrainBase):
    def __init__(self, params: Params):
        super(TrainDCGan, self).__init__(params)

    def train_step(self, epoch, batch_index, real_images):
        # Adversarial ground truths
        valid = Variable(get_tensors_type()(real_images.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(get_tensors_type()(real_images.shape[0], 1).fill_(0.0), requires_grad=False)

        self.batches_done = epoch * len(self.params.dataloader) + batch_index
        g_loss, generated_images = self.train_generator(real_images, valid)

        d_loss = self.train_discriminator(real_images, generated_images, valid, fake)

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch + 1, self.params.epochs, batch_index, len(self.params.dataloader), d_loss.item(), g_loss.item()))

    def train_generator(self, images, valid):
        self.params.generator_optimizer.zero_grad()

        # Sample noise as generator input
        noise = Variable(get_tensors_type()(np.random.normal(0, 1, (images.shape[0], self.params.latent_dim))))

        # Generate a batch of images
        generated_images = self.params.generator(noise)

        # Loss measures generator's ability to fool the discriminator
        loss = self.params.loss_function(self.params.discriminator(generated_images), valid)

        loss.backward()
        self.params.generator_optimizer.step()

        if self.batches_done % self.params.sample_interval == 0:
            save_image(generated_images.data[:25], "images/%d.png" % self.batches_done, nrow=5, normalize=True)

        return loss, generated_images

    def train_discriminator(self, real_images, generated_images, valid, fake):
        # Configure input
        real_images_as_tensor = Variable(real_images.type(get_tensors_type()))

        self.params.discriminator_optimizer.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        result_real = self.params.discriminator(real_images_as_tensor)
        result_fake = self.params.discriminator(generated_images.detach())
        real_loss = self.params.loss_function(result_real, valid)
        fake_loss = self.params.loss_function(result_fake, fake)
        loss = (real_loss + fake_loss) / 2

        loss.backward()
        self.params.discriminator_optimizer.step()
        return loss