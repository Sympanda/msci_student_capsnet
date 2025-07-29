import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---- Latent Sapce Encoding ---- #
class CapsuleLatentSpace(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(CapsuleLatentSpace, self).__init__()
        self.mean_layer = nn.Linear(input_dim, latent_dim)
        self.logvar_layer = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        return mean, logvar


# ---- Self-Attention Capsule Layer ---- #
class SelfAttentionCapsules(nn.Module):
    def __init__(self, num_input_capsules, num_output_capsules, input_dim, output_dim, attention_heads=8, dropout=0.1):
        super(SelfAttentionCapsules, self).__init__()
        self.num_input_capsules = num_input_capsules
        self.num_output_capsules = num_output_capsules
        self.output_dim = output_dim
        self.attention_heads = attention_heads
        self.head_dim = output_dim // attention_heads  

        assert self.head_dim * attention_heads == output_dim, "Output dimension must be divisible by attention_heads"

        self.query_proj = nn.Linear(input_dim, output_dim, bias=False)
        self.key_proj = nn.Linear(input_dim, output_dim, bias=False)
        self.value_proj = nn.Linear(input_dim, output_dim, bias=False)

        # Residual projection to match input to output dimensions
        self.residual_proj = nn.Linear(input_dim, output_dim, bias=False)

        self.out_proj = nn.Linear(output_dim, output_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)

        # Latent Space Encoding (mean, logvar)
        self.latent_space = CapsuleLatentSpace(output_dim, output_dim)

    def forward(self, x):
        '''
        Forward pass for self-attention routing.
        '''
        batch_size, num_input_capsules, _ = x.shape

        #assert num_input_capsules == self.num_input_capsules, "Input capsules have incorrect dimension"

        # Project Query, Key, and Value
        Q = self.query_proj(x)  # (B, num_caps, cap_dim)
        K = self.key_proj(x)    # (B, num_caps, cap_dim)
        V = self.value_proj(x)  # (B, num_caps, cap_dim)

        # Reshape for multi-head attention: (B, num_heads, num_caps, head_dim)
        Q = Q.view(batch_size, num_input_capsules, self.attention_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_input_capsules, self.attention_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_input_capsules, self.attention_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, heads, num_caps, num_caps)
        attn_weights = F.softmax(attn_scores, dim=-1)  # Normalize across capsule dimension
        attn_weights = self.dropout(attn_weights)  # Apply dropout

        # Apply attention weights
        attended_capsules = torch.matmul(attn_weights, V)  # (B, heads, num_caps, head_dim)

        # Reshape back to original shape
        attended_capsules = attended_capsules.transpose(1, 2).contiguous().view(batch_size, num_input_capsules, self.output_dim)

        # Apply output projection and residual connection
        output_capsules = self.out_proj(attended_capsules)  # (B, num_caps, cap_dim)
        projected_x = self.residual_proj(x)
        output_capsules = self.norm(output_capsules + projected_x)  # Add residual connection & normalize

        # Output shape: (batch_size, num_output_capsules, output_dim)
        # Need output shape to be (batch_size, output_dim) for routing
        attn_weights = torch.softmax(torch.mean(output_capsules, dim=-1), dim=1) # shape (batch_size, num_output_capsules)
        output_capsules = torch.sum(output_capsules * attn_weights.unsqueeze(-1), dim=1)  # shape (batch_size, output_dim)

        # Latent Space Encoding
        mean, logvar = self.latent_space(output_capsules)

        return mean, logvar