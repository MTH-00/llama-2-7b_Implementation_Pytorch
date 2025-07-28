import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # Print input dimensions
        print(f"RMSNorm input shape: {x.shape}")
        
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        normalized = self.weight * self._norm(x.float()).type_as(x)
        
        # Print output dimensions
        print(f"RMSNorm output shape: {normalized.shape}")
        
        return normalized