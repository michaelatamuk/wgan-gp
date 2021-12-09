class Params:
    def __init__(self):
        self.epochs: int = 0
        self.dataloader = None
        self.generator = None
        self.discriminator = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None
        self.gradient_penalty_lambda = None
        self.loss_function = None
        self.latent_dim = None
        self.critic = None
        self.sample_interval = None
