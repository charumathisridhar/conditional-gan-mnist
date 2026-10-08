import torch
import torch.nn as nn


# -----------------------------
# Generator
# -----------------------------
class Generator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=10, img_size=28):
        super(Generator, self).__init__()

        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.img_size = img_size

        # Convert class label into an embedding vector
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        # Generator network
        self.model = nn.Sequential(
            nn.Linear(noise_dim + num_classes, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, img_size * img_size),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Convert labels into embedding
        label_embedding = self.label_embedding(labels)

        # Combine noise and class information
        generator_input = torch.cat((noise, label_embedding), dim=1)

        # Generate image
        image = self.model(generator_input)

        # Reshape into 28 x 28 image
        image = image.view(image.size(0), 1, self.img_size, self.img_size)

        return image


# -----------------------------
# Discriminator
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self, num_classes=10, img_size=28):
        super(Discriminator, self).__init__()

        self.num_classes = num_classes
        self.img_size = img_size

        # Convert class label into an embedding
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        # Discriminator network
        self.model = nn.Sequential(
            nn.Linear(img_size * img_size + num_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, image, labels):
        # Flatten image
        image = image.view(image.size(0), -1)

        # Convert labels into embedding
        label_embedding = self.label_embedding(labels)

        # Combine image and class information
        discriminator_input = torch.cat((image, label_embedding), dim=1)

        # Determine real or fake
        validity = self.model(discriminator_input)

        return validity