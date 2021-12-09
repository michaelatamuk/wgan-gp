from torch.autograd import Variable

from train.params import Params
from train.train_dcgan_discriminator import train_discriminator
from train.train_dcgan_generator import train_generator
from utils import get_tensors_type


def train(params: Params):

    for epoch in range(params.epochs):
        for batch_index, (real_images, _) in enumerate(params.dataloader):

            # Adversarial ground truths
            valid = Variable(get_tensors_type()(real_images.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(get_tensors_type()(real_images.shape[0], 1).fill_(0.0), requires_grad=False)

            batches_done = epoch * len(params.dataloader) + batch_index
            g_loss, generated_images = train_generator(batches_done, real_images, valid, params)

            d_loss = train_discriminator(params, real_images, generated_images, valid, fake)

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch+1, params.epochs, batch_index, len(params.dataloader), d_loss.item(), g_loss.item()))

