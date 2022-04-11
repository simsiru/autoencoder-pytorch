import torch
import torch.nn as nn


# output_size = (input_size - filter_size + 2 * padding) / stride + 1
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # N, 3, 480, 480
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=1, padding=2),   # out [N, 16, 480, 480]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # out [N, 16, 240, 240]

            nn.Conv2d(16, 32, 5, stride=1, padding=2),  # out [N, 32, 240, 240]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # out [N, 32, 120, 120]

            nn.Conv2d(32, 64, 5, stride=1, padding=2),  # out [N, 64, 120, 120]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # out [N, 64, 60, 60]

            nn.Conv2d(64, 128, 5, stride=1, padding=2), # out [N, 128, 60, 60]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # out [N, 128, 30, 30]

            nn.Conv2d(128, 256, 5, stride=1, padding=2), # out [N, 128, 60, 60]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # out [N, 128, 30, 30]
        )

        # N, 128, 30, 30
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2, output_padding=1), # out [N, 64, 60, 60]
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1), # out [N, 64, 60, 60]
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 5, stride=2, padding=2, output_padding=1),  # out [N, 32, 120, 120]
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, 5, stride=2, padding=2, output_padding=1),  # out [N, 16, 240, 240]
            nn.ReLU(),

            nn.ConvTranspose2d(16, 3, 5, stride=2, padding=2, output_padding=1),   # out [N, 3, 480, 480]
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out

    def encoder_forward(self, x):
        return self.encoder(x)

    def decoder_forward(self, x):
        # N, 128, 30, 40
        return self.decoder(x)



""" from torchinfo import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConvAutoencoder().to(device)

batch_size = 1
summary(model, input_size=(batch_size, 3, 480, 640)) """

