import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from beta_loss import BetaScheduler, CyclicalBetaScheduler, kl_divergence, loss_function
from make_capsnet import build_model_from_config  # replace with your actual model path

def visualize_latent_space_effects(capsnet, test_image, latent_dim=16, steps=5, perturbation_range=2.5, save_path="latent_effects.png"):
    capsnet.eval()
    device = next(capsnet.parameters()).device
    test_image = test_image.to(device)

    with torch.no_grad():
        _, mean, logvar = capsnet(test_image.unsqueeze(0))
        mean = mean.squeeze(0)

    perturb_values = np.linspace(-perturbation_range, perturbation_range, steps)
    reconstructed_images = torch.zeros(latent_dim, steps, 28, 28)

    for dim in range(latent_dim):
        for i, perturb in enumerate(perturb_values):
            z = mean.clone()
            z[dim] += perturb
            with torch.no_grad():
                recon = capsnet.decode(z.unsqueeze(0)).squeeze(0).cpu().numpy()
            recon = (recon + 1) / 2
            reconstructed_images[dim, i] = torch.tensor(recon)

    fig, axes = plt.subplots(latent_dim, steps, figsize=(steps, latent_dim))
    for dim in range(latent_dim):
        for i in range(steps):
            axes[dim, i].imshow(reconstructed_images[dim, i], cmap="gray")
            axes[dim, i].axis("off")
    plt.tight_layout(pad=0.1)
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved visualization at {save_path}")

def train_mnist_capsnet(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = MNIST(root="./data", train=True, transform=transform, download=True)
    test_ds = MNIST(root="./data", train=False, transform=transform, download=True)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

    # Model
    model = build_model_from_config(config).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 1e-5))

    # Scheduler
    beta_scheduler = BetaScheduler(start_beta=config["beta_start"], max_beta=config["beta_max"], steps=config["beta_steps"])

    def custom_loss(x, x_recon, mean, logvar, beta):
        x = (x + 1) / 2
        x_recon = (x_recon + 1) / 2
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        total_loss, kl_loss = loss_function(recon_loss, mean, logvar, beta)
        return recon_loss, total_loss, kl_loss

    # Train
    model.train()
    step = 0
    for epoch in range(config["epochs"]):
        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x = x.to(device)
            optimizer.zero_grad()
            recon, mean, logvar = model(x)
            beta = beta_scheduler.step()
            recon_loss, total_loss, kl_loss = custom_loss(x, recon, mean, logvar, beta)
            total_loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"[{epoch+1}/{config['epochs']}] Step {step} | Recon: {recon_loss.item():.2f}, KL: {kl_loss.item():.2f}, Total: {total_loss.item():.2f}")
            step += 1

    # Save model
    torch.save(model.state_dict(), config.get("model_save_path", "capsnet_beta_mnist.pth"))

    # Evaluate and plot reconstructions
    model.eval()
    with torch.no_grad():
        test_images, _ = next(iter(test_loader))
        test_images = test_images[:8].to(device)
        recons, _, _ = model(test_images)

    fig, axes = plt.subplots(2, 8, figsize=(10, 3))
    for i in range(8):
        axes[0, i].imshow(test_images[i].cpu().squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recons[i].cpu().squeeze(), cmap="gray")
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.show()

    # Latent effect visualization
    visualize_latent_space_effects(
        model,
        test_images[0].cpu(),
        latent_dim=config["latent_dim"],
        steps=5,
        save_path="mnist_latent_effects.png"
    )

import h5py

class GalaxyZooH5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            self.x = f['images'][:]  # shape [N, 128, 128, 3]
        self.x = torch.tensor(self.x).permute(0, 3, 1, 2).float()
        self.x = (self.x / 255.0 - 0.5) * 2  # normalize to [-1, 1]

    def __len__(self): return self.x.shape[0]
    def __getitem__(self, i): return self.x[i]

def train_galaxy_capsnet(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = GalaxyZooH5Dataset(config["train_data_path"])
    test_ds = GalaxyZooH5Dataset(config["test_data_path"])
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

    model = build_model_from_config(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 1e-5))
    beta_scheduler = BetaScheduler(start_beta=config["beta_start"], max_beta=config["beta_max"], steps=config["beta_steps"])

    def custom_loss(x, x_recon, mean, logvar, beta):
        x = (x + 1) / 2
        x_recon = (x_recon + 1) / 2
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        total_loss, kl_loss = loss_function(recon_loss, mean, logvar, beta)
        return recon_loss, total_loss, kl_loss

    step = 0
    for epoch in range(config["epochs"]):
        for x in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x = x.to(device)
            print(x.shape)
            optimizer.zero_grad()
            recon, mean, logvar = model(x)
            beta = beta_scheduler.step()
            recon_loss, total_loss, kl_loss = custom_loss(x, recon, mean, logvar, beta)
            total_loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"[{epoch+1}/{config['epochs']}] Step {step} | Recon: {recon_loss.item():.2f}, KL: {kl_loss.item():.2f}, Total: {total_loss.item():.2f}")
            step += 1

    torch.save(model.state_dict(), config.get("model_save_path", "capsnet_galaxy.pth"))

    # Evaluate + plot RGB reconstructions
    model.eval()
    with torch.no_grad():
        test_images = next(iter(test_loader))[:8].to(device)
        recons, _, _ = model(test_images)

    fig, axes = plt.subplots(2, 8, figsize=(12, 3))
    for i in range(8):
        axes[0, i].imshow(test_images[i].cpu().permute(1, 2, 0).clamp(-1, 1).add(1).div(2).numpy())
        axes[0, i].axis("off")
        axes[1, i].imshow(recons[i].cpu().permute(1, 2, 0).clamp(-1, 1).add(1).div(2).numpy())
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.show()

    # Latent visualization
    visualize_latent_space_effects(
        model,
        test_images[0].cpu(),
        latent_dim=config["latent_dim"],
        steps=5,
        save_path="galaxy_latent_effects.png"
    )