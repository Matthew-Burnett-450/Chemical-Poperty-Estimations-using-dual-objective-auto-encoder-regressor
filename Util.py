import numpy as np
import torch
class MinMaxScaler():
    def __init__(self):
        self.min=None
        self.max=None
    def fit(self,X):
        self.min=np.min(X)
        self.max=np.max(X)
    def transform(self,X):
        return (X-self.min)/(self.max-self.min)
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)
    def inverse_transform(self,X):
        return X*(self.max-self.min)+self.min
    
def sample_latent_space(mu, log_var, num_samples=10):
    """
    Generate samples from the latent space distribution defined by mu and log_var.
    Args:
        mu (Tensor): Mean of the latent space distribution.
        log_var (Tensor): Log variance of the latent space distribution.
        num_samples (int): Number of samples to generate.
    Returns:
        List[Tensor]: A list of sampled latent vectors.
    """
    std = torch.exp(0.5 * log_var)
    samples = [mu + std * torch.randn_like(std) for _ in range(num_samples)]
    return samples
