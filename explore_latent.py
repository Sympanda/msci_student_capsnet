import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from feature_extractor import FeatureExtractor
from primary_caps import PrimaryCapsules
from mnist_test_v2 import BetaCapsNet, CNNDecoder

def visualize_latent_space_effects(capsnet, test_image, latent_dim=16, steps=5, perturbation_range=2.5, save_path="latent_effects.png"):
    """
    Visualizes how changing each latent dimension affects the reconstruction.
    
    Parameters:
    - capsnet: The trained Capsule Network (with decoder).
    - test_image: A single MNIST image tensor of shape [1, 1, 28, 28].
    - latent_dim: Number of latent dimensions.
    - steps: Number of variations for each dimension (default: 5).
    - perturbation_range: Maximum change applied to latent dimensions (default: ±2.5).
    - save_path: Where to save the figure.
    """
    
    capsnet.eval()  # Set model to evaluation mode
    
    # Move input to correct device
    device = next(capsnet.parameters()).device
    test_image = test_image.to(device)

    with torch.no_grad():
        # Encode test image to get latent mean & logvar
        _, mean, logvar = capsnet(test_image.unsqueeze(0))  # Add batch dim
        mean = mean.squeeze(0)  # Remove batch dim
    
    # Define perturbation levels
    perturb_values = np.linspace(-perturbation_range, perturbation_range, steps)
    
    # Create grid to store generated images
    reconstructed_images = torch.zeros(latent_dim, steps, 28, 28)

    for dim in range(latent_dim):  # Iterate over each latent dim
        for i, perturb in enumerate(perturb_values):
            perturbed_latent = mean.clone()  # Copy original latent vector
            perturbed_latent[dim] += perturb  # Modify one dimension
            
            with torch.no_grad():
                recon_image = capsnet.decode(perturbed_latent.unsqueeze(0)).squeeze(0).cpu().numpy()
            
            # Rescale from [-1,1] to [0,1] for visualization
            recon_image = (recon_image + 1) / 2  
            reconstructed_images[dim, i] = torch.tensor(recon_image)

    # Create figure
    fig, axes = plt.subplots(latent_dim, steps, figsize=(steps, latent_dim))

    for dim in range(latent_dim):
        for i in range(steps):
            axes[dim, i].imshow(reconstructed_images[dim, i], cmap="gray")
            axes[dim, i].axis("off")

    # Save & Show
    plt.tight_layout(pad=0.1)
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Saved visualization at {save_path}")



# Load dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load trained Capsule Network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

capsnet = BetaCapsNet(
    in_channels=1,
    feature_dim=256,
    num_primary_capsules=28,
    primary_caps_dim=8,
    output_caps_dim=16
).to(device)

# Load the trained weights
capsnet.load_state_dict(torch.load("unsupervised_CapsNet/capsnet_beta_v3.pth", map_location=torch.device("cpu")))


# Load a sample image from MNIST
capsnet.eval()
with torch.no_grad():
    test_image, _ = next(iter(test_loader))  # Get a batch
    test_image = test_image[4]  # Take a single image

visualize_latent_space_effects(capsnet, test_image, latent_dim=16, steps=5, perturbation_range=2.5, save_path="latent_effects.png")

