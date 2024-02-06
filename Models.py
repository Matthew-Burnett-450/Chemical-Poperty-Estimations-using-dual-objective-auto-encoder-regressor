import torch
import torch.nn as nn
import torch.nn.functional as F



    
#novel model

class SemiSupervisedModel(nn.Module):
    def __init__(self, input_dim, latent_dim,output_dim):
        super(SemiSupervisedModel, self).__init__()
        # Define the encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),  # First reduction step
            nn.ReLU(),
            nn.Linear(1024, 512),  # Further reduction
            nn.ReLU(),
            nn.Linear(512, 256),  # Further reduction
            nn.ReLU(),
            nn.Linear(256, 128),  # Further reduction
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        # Define the decoder with a gradual expansion
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),  # Start expanding from latent_dim
            nn.ReLU(),
            nn.Linear(128, 256),  # Further expansion
            nn.ReLU(),
            nn.Linear(256, 512),  # Further expansion
            nn.ReLU(),
            nn.Linear(512, 1024),  # Further expansion
            nn.ReLU(),
            nn.Linear(1024, input_dim),  # Back to original dimension
            nn.Sigmoid(),
        )
        # Regression head for two targets
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, 15),  # Increase from the original size
            nn.BatchNorm1d(15),
            nn.Sigmoid(),
            nn.Linear(15,15),  # Additional layer
            nn.BatchNorm1d(15),
            nn.Sigmoid(),
            nn.Linear(15, output_dim),  # Output dimension for regression targets
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        regression_output = self.regressor(encoded)
        return decoded, regression_output
