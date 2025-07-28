import torch
import torch.nn as nn
from dataclasses import dataclass
from attention.attention import SelfAttention
from ffn.ffn import FeedForward
from norm.norm import RMSNorm
from typing import Optional
from attention.attention import ModelArgs
# @dataclass
# class ModelArgs:
#     dim: int
#     n_heads: int
#     n_kv_heads: int = None
#     multiple_of: int = 256
#     ffn_dim_multiplier: float = None
#     norm_eps: float = 1e-5
#     # max_batch_size: int = 32
#     # ffn_dim_multiplier: float = 4.0
#     # max_seq_len: int = 2048


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

class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.has_printed = False
        
        
        if not self.has_printed:
            print(f"\nInitializing EncoderBlock:")
            print(f"Model dimension: {args.dim}")
            print(f"Number of heads: {args.n_heads}")
            print(f"Number of KV heads: {args.n_kv_heads if args.n_kv_heads else args.n_heads}")
            print(f"Normalization epsilon: {args.norm_eps}")
            self.has_printed = True
        # Initialize components
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        
        # Initialize normalization layers
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        print(f"\nEncoderBlock Forward Pass:")
        print(f"Input shape: {x.shape}")
        
        # First residual block: Attention
        h = x
        print(f"\nPre-attention normalization:")
        h = self.attention_norm(h)
        print(f"After attention norm shape: {h.shape}")
        
        print(f"\nApplying self-attention:")
        h = self.attention(h, start_pos, freqs_complex)
        print(f"After attention shape: {h.shape}")
        
        x = x + h
        print(f"After first residual connection: {x.shape}")
        
        # Second residual block: Feed-forward
        h = x
        print(f"\nPre-FFN normalization:")
        h = self.ffn_norm(h)
        print(f"After FFN norm shape: {h.shape}")
        
        print(f"\nApplying feed-forward:")
        h = self.feed_forward(h)
        print(f"After feed-forward shape: {h.shape}")
        
        x = x + h
        print(f"After second residual connection: {x.shape}")
        
        return x