
import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(7, 16, kernel_size=3, stride=1, padding=1),  # (N, 32, 120, 120)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (N, 32, 60, 60)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (N, 64, 60, 60)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (N, 64, 30, 30)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (N, 128, 30, 30)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (N, 128, 15, 15)
            nn.Flatten(),  # Flatten the features
            nn.Linear(64 * 15 * 15, 64)  # Latent space representation
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 64 * 15 * 15),
            nn.Unflatten(1, (64, 15, 15)),  # Unflatten to match the encoder's last conv layer
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 7, kernel_size=2, stride=2),
            nn.Sigmoid()  # Ensuring output values are between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        return self.encoder(x)