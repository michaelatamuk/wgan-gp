from train.params import Params
from train.train_wgan_discriminator import train_discriminator
from train.train_wgan_generator import train_generator


def train(params: Params):
    batches_done = 0
    for epoch in range(params.epochs):
        for batch_index, (imgs, _) in enumerate(params.dataloader):

            discriminator_loss, noise = train_discriminator(imgs, params)

            # Train the generator every critic steps
            if batch_index % params.critic == 0:
                generator_loss = train_generator(batches_done, params, noise)
                batches_done += params.critic

                print("[Epoch %d/%d] [Batch %d/%d] [Discriminator loss: %f] [Generator loss: %f]"
                      % (epoch+1, params.epochs, batch_index, len(params.dataloader),
                         discriminator_loss.item(), generator_loss.item()))

