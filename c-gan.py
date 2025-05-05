"""
22035587, c-gan.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn.utils.spectral_norm import spectral_norm

# Define the device to be used (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a custom Dataset class
class Kuzushiji49Dataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.images = np.load(images_path)['arr_0']
        self.labels = np.load(labels_path)['arr_0']
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Load the dataset
train_dataset = Kuzushiji49Dataset('k49-train-imgs.npz', 'k49-train-labels.npz', transform=lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(0) / 255.0)
test_dataset = Kuzushiji49Dataset('k49-test-imgs.npz', 'k49-test-labels.npz', transform=lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(0) / 255.0)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Utility function for one-hot encoding
def to_one_hot(labels, num_classes):
    return F.one_hot(labels.long(), num_classes).float()

# Constants
num_classes = 49
image_size = 28 * 28
latent_dim = 100

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_dim + num_classes, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, image_size)

    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return torch.sigmoid(self.fc3(x))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = spectral_norm(nn.Linear(image_size + num_classes, 512))
        self.fc2 = spectral_norm(nn.Linear(512, 256))
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, c):
        x = torch.cat([x, c], dim=1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        return torch.sigmoid(self.fc3(x))

# Instantiate models, optimizers, and loss function
generator = Generator().to(device)
discriminator = Discriminator().to(device)
optimizer_G = optim.Adam(generator.parameters(), lr=1e-4)
optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4)
adversarial_loss = nn.BCELoss()

# Training loop for C-GAN
def train_cgan(epoch):
    generator.train()
    discriminator.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        batch_size = data.size(0)
        data, labels = data.to(device), labels.to(device)
        one_hot_labels = to_one_hot(labels, num_classes).to(device)
        
        # Adversarial ground truths
        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)
        
        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim, device=device)
        gen_labels = to_one_hot(torch.randint(0, num_classes, (batch_size,), device=device), num_classes)
        gen_images = generator(z, gen_labels)
        g_loss = adversarial_loss(discriminator(gen_images, gen_labels), valid)
        g_loss.backward()
        optimizer_G.step()
        
        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(data.view(-1, image_size), one_hot_labels), valid)
        fake_loss = adversarial_loss(discriminator(gen_images.detach(), gen_labels), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

# Testing routine to generate samples
def generate_cgan_samples(label, num_samples=49):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        one_hot_label = to_one_hot(torch.tensor([label] * num_samples, device=device), num_classes).to(device)
        samples = generator(z, one_hot_label).cpu()
        return samples.view(num_samples, 1, 28, 28)

# Training the C-GAN
for epoch in range(1, 51):
    train_cgan(epoch)

# Generate and visualize samples for a specific class (e.g., class 0)
samples = generate_cgan_samples(0)

# Plotting the generated samples in a 7x7 grid
def plot_images(images, title):
    fig, axes = plt.subplots(7, 7, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i].squeeze(), cmap='gray')
        ax.axis('off')
    fig.suptitle(title, fontsize=20)
    plt.show()

plot_images(samples, 'Generated Samples for Class 0')
