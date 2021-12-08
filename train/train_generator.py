from torchvision.utils import save_image
import torch


def train_generator(batches_done, d_loss, epoch, i, params, noise):
    # Generate a batch of images
    fake_imgs = params.generator(noise)

    # Loss measures generator's ability to fool the discriminator
    # Train on fake images
    fake_validity = params.discriminator(fake_imgs)
    g_loss = -torch.mean(fake_validity)
    g_loss.backward()
    params.generator_optimizer.step()
    print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
          % (epoch, params.epochs, i, len(params.dataloader), d_loss.item(), g_loss.item()))

    if batches_done % params.sample_interval == 0:
        save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
    batches_done += params.critic
