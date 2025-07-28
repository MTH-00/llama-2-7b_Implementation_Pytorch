import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from attention.attention import ModelArgs


# @dataclass
# class ModelArgs:
#     dim: int
#     multiple_of: int = 256
#     ffn_dim_multiplier: float = None


# @dataclass
# class ModelArgs:
#     dim: int
#     n_heads: int
#     n_kv_heads: int = None
#     multiple_of: int = 256
#     ffn_dim_multiplier: float = None
#     norm_eps: float = 1e-5
#     max_batch_size: int = 32
#     max_seq_len: int = 2048
#     vocab_size: int = -1
#     device: Optional[str] = None



class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()  
        self.has_printed = False

        # Calculate hidden dimension
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        
        if not self.has_printed:
            print(f"\nInitializing FeedForward Network:")
            print(f"Input dimension: {args.dim}")
            print(f"Initial hidden dimension: {hidden_dim}")
            self.has_printed = True
        # Apply FFN dimension multiplier if provided
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
            print(f"After multiplier hidden dimension: {hidden_dim}")
        
        # Round the hidden_dim to the nearest multiple of multiple_of
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        print(f"Final hidden dimension (rounded): {hidden_dim}")

        # Initialize linear layers
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
        
        print(f"W1 shape: {args.dim} -> {hidden_dim}")
        print(f"W2 shape: {hidden_dim} -> {args.dim}")
        print(f"W3 shape: {args.dim} -> {hidden_dim}")

    def forward(self, x: torch.Tensor):
        print(f"\nFeedForward Forward Pass:")
        print(f"Input shape: {x.shape}")
        
        # First projection and SwiGLU activation
        swish = F.silu(self.w1(x))
        print(f"After W1 and SiLU shape: {swish.shape}")
        
        # Second projection
        x_V = self.w3(x)
        print(f"After W3 shape: {x_V.shape}")
        
        # Elementwise multiplication
        x = swish * x_V
        print(f"After multiplication shape: {x.shape}")
        
        # Final projection
        x = self.w2(x)
        print(f"Final output shape After FFN: {x.shape}")
        
        return x