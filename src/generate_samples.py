import torch
import matplotlib.pyplot as plt
from model import Generator


# -----------------------------
# Settings
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

noise_dim = 100
num_classes = 10
img_size = 28

model_path = "models/generator.pth"


# -----------------------------
# Load Generator
# -----------------------------
generator = Generator(
    noise_dim=noise_dim,
    num_classes=num_classes,
    img_size=img_size
).to(device)

generator.load_state_dict(torch.load(model_path, map_location=device))

generator.eval()

print("Generator model loaded successfully!")


# -----------------------------
# Generate samples
# -----------------------------
num_samples = 10

# Random noise
noise = torch.randn(num_samples, noise_dim).to(device)

# Class labels: 0 to 9
labels = torch.arange(0, num_classes).to(device)

# Generate images
with torch.no_grad():
    generated_images = generator(noise, labels)


# -----------------------------
# Display generated images
# -----------------------------
plt.figure(figsize=(15, 3))

for i in range(num_samples):
    plt.subplot(2, 5, i + 1)

    image = generated_images[i].cpu().squeeze()

    plt.imshow(image, cmap="gray")
    plt.title(f"Class {i}")
    plt.axis("off")

plt.tight_layout()

# Save result
plt.savefig(
    "results/screenshots/conditional_generated_samples.png",
    dpi=150
)

plt.show()

print("Generated images saved successfully!")
print("Location: results/screenshots/conditional_generated_samples.png")