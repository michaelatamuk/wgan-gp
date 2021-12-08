from train.params import Params
from train.train_discriminator import train_discriminator
from train.train_generator import train_generator


def train(params: Params):
    batches_done = 0
    for epoch in range(params.epochs):
        for i, (imgs, _) in enumerate(params.dataloader):

            d_loss, noise = train_discriminator(imgs, params)

            # Train the generator every critic steps
            if i % params.critic == 0:
                train_generator(batches_done, d_loss, epoch, i, params, noise)
