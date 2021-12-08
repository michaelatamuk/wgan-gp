import torch
from torchvision.utils import save_image


def train_generator(batches_done, params, noise):
    # Generate a batch of images
    fake_images = params.generator(noise)

    # Loss measures generator's ability to fool the discriminator
    # Train on fake images
    fake_validity = params.discriminator(fake_images)
    loss = -torch.mean(fake_validity)
    loss.backward()
    params.generator_optimizer.step()

    if batches_done % params.sample_interval == 0:
        save_image(fake_images.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

    batches_done += params.critic

    return loss
