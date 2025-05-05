"""
22035587, c-vae.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.encoder_fc1 = nn.Linear(image_size + num_classes, 512)
        self.encoder_fc2_mu = nn.Linear(512, latent_dim)
        self.encoder_fc2_logvar = nn.Linear(512, latent_dim)
        self.decoder_fc1 = nn.Linear(latent_dim + num_classes, 512)
        self.decoder_fc2 = nn.Linear(512, image_size)

    def encode(self, x, c):
        x = torch.cat([x, c], dim=1)
        h = F.relu(self.encoder_fc1(x))
        mu = self.encoder_fc2_mu(h)
        logvar = self.encoder_fc2_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        z = torch.cat([z, c], dim=1)
        h = F.relu(self.decoder_fc1(z))
        return torch.sigmoid(self.decoder_fc2(h))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, image_size), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Instantiate model, optimizer, and loss function
cvae = CVAE().to(device)
optimizer = optim.Adam(cvae.parameters(), lr=1e-3)

# Training loop for C-VAE
def train_cvae(epoch):
    cvae.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        one_hot_labels = to_one_hot(labels, num_classes).to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = cvae(data.view(-1, image_size), one_hot_labels)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item() / len(data)}')

# Testing routine to generate samples
def generate_cvae_samples(label, num_samples=49):
    cvae.eval()
    with torch.no_grad():
        one_hot_label = to_one_hot(torch.tensor([label] * num_samples), num_classes).to(device)
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = cvae.decode(z, one_hot_label).cpu()
        return samples.view(num_samples, 1, 28, 28)

# Training the C-VAE
for epoch in range(1, 21):
    train_cvae(epoch)

# Generate and visualize samples for a specific class (e.g., class 0)
samples = generate_cvae_samples(0)

# Plotting the generated samples in a 7x7 grid
def plot_images(images, title):
    num_images = len(images)
    grid_size = 7  # Since we want a 7x7 grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(images[i].squeeze(), cmap='gray')
        ax.axis('off')
    fig.suptitle(title, fontsize=20)
    plt.show()

plot_images(samples, 'Generated Samples for Class 0')
