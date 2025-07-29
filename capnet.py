import torch
import torch.nn as nn
import torch.nn.functional as F
from feature_extractor import FeatureExtractor
from primary_caps import PrimaryCapsules
from attention import SelfAttentionCapsules
from torchsummary import summary


# ---- Full Capsule Network ---- #
class CapsuleNetwork(nn.Module):
    def __init__(self, in_channels, feature_dim, num_primary_capsules, primary_caps_dim, output_caps_dim):
        super(CapsuleNetwork, self).__init__()

        # Feature Extractor (Replaceable with ResNet, etc.)
        self.feature_extractor = FeatureExtractor(in_channels=in_channels, feature_dim=feature_dim)

        # Primary Capsule Layer
        self.primary_caps = PrimaryCapsules(
            in_channels=feature_dim,  
            num_capsules=num_primary_capsules,  
            capsule_dim=primary_caps_dim,  
            kernel_size=7,  
            stride=2  
        )

        # Self-Attention Capsule Routing to enforce disentanglement
        self.attention_caps = SelfAttentionCapsules(
            num_input_capsules=num_primary_capsules * 6 * 6,  # Adjust for actual output shape
            num_output_capsules=1,  # **Force a single capsule for 1x16 latent space**
            input_dim=primary_caps_dim,  
            output_dim=output_caps_dim,
            attention_heads=8,
        )

    def forward(self, x):
        features = self.feature_extractor(x)  
        primary_caps_output = self.primary_caps(features)  
        mean, logvar = self.attention_caps(primary_caps_output)
        #print(f"Primary Capsules Output Shape: {primary_caps_output.shape}")

        return mean, logvar  # Output shape: (batch_size, 16)


# ---- Testing the Updated Network ---- #
if __name__ == "__main__":
    print("Testing Capsule Network with Self-Attention Routing...")
    
    batch_size = 32
    input_tensor = torch.randn(batch_size, 3, 28, 28)  # Example grayscale galaxy images (1 channel, 64x64)
    
    capsnet = CapsuleNetwork(
        in_channels=3, 
        feature_dim=256,  
        num_primary_capsules=36,  
        primary_caps_dim=8,  
        output_caps_dim=16  
    )

    output, output_logvar = capsnet(input_tensor)
    summary(capsnet, input_size=(3, 28, 28))
    print(f"Final Latent Space Output Shape: {output.shape}")  # Expect (batch_size, 16)

    ### Check fake forward pass
    with torch.no_grad():
        test_input = torch.randn(4, 3, 28, 28) # small batch size
        test_output, _ = capsnet(test_input)
        print(f"Test Output Shape: {test_output.shape}")

    ### Ensure Gradients Flow Properly
    optimizer = torch.optim.Adam(capsnet.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    test_input = torch.randn(4, 3, 28, 28)
    target_output = torch.randn(4, 16)

    optimizer.zero_grad()
    output, _ = capsnet(test_input)
    loss = loss_fn(output, target_output)
    loss.backward()
    optimizer.step()
    print("Gradient Check Passed!")