import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from model import Generator, Discriminator


# ============================================================
# Project Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 64

# Increased from 20 to 50 for better image quality
EPOCHS = 50

NOISE_DIM = 100
NUM_CLASSES = 10

LEARNING_RATE = 0.0002
IMAGE_SIZE = 28

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("======================================")
print("Conditional GAN Training")
print("======================================")
print("Using device:", DEVICE)
print("Epochs:", EPOCHS)


# ============================================================
# MNIST Dataset
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print("Dataset loaded successfully!")
print("Training images:", len(dataset))


# ============================================================
# Generator
# ============================================================

generator = Generator(
    noise_dim=NOISE_DIM,
    num_classes=NUM_CLASSES,
    img_size=IMAGE_SIZE
).to(DEVICE)


# ============================================================
# Discriminator
# ============================================================

discriminator = Discriminator(
    num_classes=NUM_CLASSES,
    img_size=IMAGE_SIZE
).to(DEVICE)


# ============================================================
# Loss
# ============================================================

criterion = nn.BCELoss()


# ============================================================
# Optimizers
# ============================================================

optimizer_G = optim.Adam(
    generator.parameters(),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999)
)

optimizer_D = optim.Adam(
    discriminator.parameters(),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999)
)


# ============================================================
# Store Losses
# ============================================================

generator_losses = []
discriminator_losses = []


# ============================================================
# Fixed Noise
# ============================================================

# Same noise is used after every epoch.
# This helps us compare how image quality improves.

fixed_noise = torch.randn(
    NUM_CLASSES,
    NOISE_DIM,
    device=DEVICE
)

fixed_labels = torch.arange(
    NUM_CLASSES,
    device=DEVICE
)


# ============================================================
# Training
# ============================================================

print()
print("Starting Conditional GAN training...")
print()

for epoch in range(EPOCHS):

    total_g_loss = 0
    total_d_loss = 0

    for batch_idx, (real_images, labels) in enumerate(dataloader):

        real_images = real_images.to(DEVICE)
        labels = labels.to(DEVICE)

        batch_size = real_images.size(0)

        # ----------------------------------------------------
        # Real and Fake labels
        # ----------------------------------------------------

        # Slight label smoothing helps GAN training
        real_targets = torch.full(
            (batch_size, 1),
            0.9,
            device=DEVICE
        )

        fake_targets = torch.zeros(
            batch_size,
            1,
            device=DEVICE
        )

        # ====================================================
        # Train Discriminator
        # ====================================================

        optimizer_D.zero_grad()

        # Real images
        real_output = discriminator(
            real_images,
            labels
        )

        real_loss = criterion(
            real_output,
            real_targets
        )

        # Generate fake images
        noise = torch.randn(
            batch_size,
            NOISE_DIM,
            device=DEVICE
        )

        fake_images = generator(
            noise,
            labels
        )

        # Discriminator prediction for fake images
        fake_output = discriminator(
            fake_images.detach(),
            labels
        )

        fake_loss = criterion(
            fake_output,
            fake_targets
        )

        # Total discriminator loss
        discriminator_loss = (
            real_loss + fake_loss
        )

        discriminator_loss.backward()
        optimizer_D.step()

        # ====================================================
        # Train Generator
        # ====================================================

        optimizer_G.zero_grad()

        noise = torch.randn(
            batch_size,
            NOISE_DIM,
            device=DEVICE
        )

        generated_images = generator(
            noise,
            labels
        )

        output = discriminator(
            generated_images,
            labels
        )

        # Generator wants discriminator
        # to classify fake images as real
        generator_loss = criterion(
            output,
            real_targets
        )

        generator_loss.backward()
        optimizer_G.step()

        total_g_loss += generator_loss.item()
        total_d_loss += discriminator_loss.item()

    # ========================================================
    # Average Loss
    # ========================================================

    avg_g_loss = (
        total_g_loss / len(dataloader)
    )

    avg_d_loss = (
        total_d_loss / len(dataloader)
    )

    generator_losses.append(avg_g_loss)
    discriminator_losses.append(avg_d_loss)

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Generator Loss: {avg_g_loss:.4f} | "
        f"Discriminator Loss: {avg_d_loss:.4f}"
    )

    # ========================================================
    # Generate Sample Every 5 Epochs
    # ========================================================

    if (epoch + 1) % 5 == 0:

        generator.eval()

        with torch.no_grad():

            sample_images = generator(
                fixed_noise,
                fixed_labels
            )

        generator.train()

        # Create figure
        plt.figure(figsize=(15, 3))

        for i in range(NUM_CLASSES):

            plt.subplot(2, 5, i + 1)

            image = (
                sample_images[i]
                .cpu()
                .squeeze()
            )

            plt.imshow(
                image,
                cmap="gray"
            )

            plt.title(
                f"Class {i}"
            )

            plt.axis("off")

        plt.tight_layout()

        sample_path = os.path.join(
            RESULTS_DIR,
            f"sample_epoch_{epoch + 1}.png"
        )

        plt.savefig(
            sample_path,
            dpi=150
        )

        plt.close()

        print(
            f"Sample saved: sample_epoch_{epoch + 1}.png"
        )


# ============================================================
# Save Models
# ============================================================

generator_path = os.path.join(
    MODELS_DIR,
    "generator.pth"
)

discriminator_path = os.path.join(
    MODELS_DIR,
    "discriminator.pth"
)

torch.save(
    generator.state_dict(),
    generator_path
)

torch.save(
    discriminator.state_dict(),
    discriminator_path
)


# ============================================================
# Plot Training Loss
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    generator_losses,
    label="Generator Loss"
)

plt.plot(
    discriminator_losses,
    label="Discriminator Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Conditional GAN Training Loss"
)

plt.legend()
plt.grid(True)

loss_path = os.path.join(
    RESULTS_DIR,
    "training_loss.png"
)

plt.savefig(
    loss_path,
    dpi=150
)

plt.close()


# ============================================================
# Finished
# ============================================================

print()
print("======================================")
print("Training completed successfully!")
print("======================================")

print("Generator saved at:")
print(generator_path)

print()
print("Discriminator saved at:")
print(discriminator_path)

print()
print("Loss graph saved at:")
print(loss_path)

print()
print("Sample images are saved in:")
print(RESULTS_DIR)