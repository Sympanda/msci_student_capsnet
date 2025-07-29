import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Primary Capsules Layer ---- #
class PrimaryCapsules(nn.Module):
    def __init__(self, in_channels, num_capsules, capsule_dim, kernel_size, stride):
        super(PrimaryCapsules, self).__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.conv = nn.Conv2d(in_channels, num_capsules * capsule_dim, kernel_size=kernel_size, stride=stride)
    
    def forward(self, x):
        out = self.conv(x)  
        batch_size = x.size(0)
        h, w = out.size(2), out.size(3)
        out = out.view(batch_size, self.num_capsules, self.capsule_dim, h * w)
        out = out.permute(0, 3, 1, 2).contiguous()  
        out = out.view(batch_size, h * w * self.num_capsules, self.capsule_dim)  
        return self.squash(out)
    
    def squash(self, x):
        norm = (x ** 2).sum(dim=2, keepdim=True)
        scale = norm / (1 + norm) / torch.sqrt(norm + 1e-9)
        return scale * x

