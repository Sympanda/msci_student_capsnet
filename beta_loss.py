import numpy as np
import torch


class BetaScheduler:
    def __init__(self, start_beta=0.001, max_beta=1.0, steps=10000):
        self.start_beta = start_beta
        self.max_beta = max_beta
        self.steps = steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        beta = self.start_beta + (self.max_beta - self.start_beta) * min(1.0, self.current_step / self.steps)
        return beta



class CyclicalBetaScheduler:
    def __init__(self, start_beta=0.001, max_beta=1.0, cycle_length=5000, total_steps=50000):
        self.start_beta = start_beta
        self.max_beta = max_beta
        self.cycle_length = cycle_length
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        cycle_position = (self.current_step % self.cycle_length) / self.cycle_length
        beta = self.start_beta + (self.max_beta - self.start_beta) * np.sin(cycle_position * np.pi)
        return beta

def kl_divergence(mean, logvar):
    """Compute KL divergence between q(z|x) and p(z) (assumed unit Gaussian)."""
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1)

def loss_function(recon_loss, mean, logvar, beta):
    """Total loss with reconstruction + β-weighted KL divergence."""
    kl_loss = kl_divergence(mean, logvar).mean()
    return recon_loss + beta * kl_loss, kl_loss
