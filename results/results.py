from abc import ABC, abstractmethod

import torch


class Results:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.last_generator: torch.nn.Module = None
        self.generator_losses: {} = {}
        self.discriminator_losses: {} = {}

    @abstractmethod
    def get_model_name(self):
        pass

    def loss_updated_callback(self, step_num: int, generator: torch.nn.Module, generator_loss: float,
                              discriminator_loss: float):
        self.last_generator = generator
        torch.save(self.last_generator, "results/" + self.model_name + ".pt")
        torch.save(self.last_generator.state_dict(), "results/" + self.model_name + "_state_dict.pt")

        self.generator_losses[step_num] = generator_loss
        self.discriminator_losses[step_num] = discriminator_loss
