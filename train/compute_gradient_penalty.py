import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd

from utils import get_tensors_type


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculate the gradient penalty loss for WGAN GP"""

    # Random weight term for interpolation between real and fake samples
    alpha = get_tensors_type()(np.random.random((real_samples.size(0), 1, 1, 1)))

    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(get_tensors_type()(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)

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
