import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from rope.rope import apply_rotary_embeddings
from typing import Optional


# @dataclass
# class ModelArgs:
#     dim: int
#     n_heads: int
#     n_kv_heads: int = None
#     max_batch_size: int = 32
#     max_seq_len: int = 2048
@dataclass
class ModelArgs:
    dim: int
    n_layers: int = 1
    n_heads: int = 1 #------> 12 for llama do research
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: Optional[str] = None
    dropout: float = 0.1   




def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads to match the number of query heads."""
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    print(f"\nRepeating KV heads:")
    print(f"Input shape: {x.shape}")
    x = x[:, :, :, None, :].expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
    x = x.reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    print(f"Output shape after repeating: {x.shape}")
    return x

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.has_printed = False

        # # Assertions for model arguments
        # assert args.vocab_size != -1, "Vocab size must be set"
        # assert args.n_kv_heads is None or args.n_kv_heads > 0, "Number of key/value heads must be a positive integer or None"
        # assert args.max_batch_size > 0, "Max batch size must be a positive integer"
        # assert args.max_seq_len > 0, "Max sequence length must be a positive integer"
        # assert args.dim % args.n_heads == 0, "Model dimension must be divisible by the number of heads"
        if not self.has_printed:
            print(f"\nInitializing SelfAttention:")
            print(f"Model dimension: {args.dim}")
            print(f"Number of query heads: {args.n_heads}")
            print(f"Number of key/value heads: {args.n_kv_heads if args.n_kv_heads else args.n_heads}")
            self.has_printed = True

        # Attention head dimensions
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        # Linear projections
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # KV cache
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        print(f"\nSelfAttention Forward Pass:")
        print(f"Input shape: {x.shape}")

        batch_size, seq_len, _ = x.shape

        # Linear projections with shape tracking
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        print(f"After linear projections - Q: {xq.shape}, K: {xk.shape}, V: {xv.shape}")

        # Reshape for attention heads
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        print(f"After reshaping - Q: {xq.shape}, K: {xk.shape}, V: {xv.shape}")

        # # Apply rotary embeddings
        # xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        # xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)
        # print(f"After RoPE - Q: {xq.shape}, K: {xk.shape}")

        # Apply rotary embeddings
        xq = apply_rotary_embeddings(xq, freqs_complex.type_as(x), device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex.type_as(x), device=x.device)
        print(f"After RoPE - Q: {xq.shape}, K: {xk.shape}, V: {xv.shape}")

        # # Update KV cache
        # self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        # self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        # # Get keys and values from cache
        # keys = self.cache_k[:batch_size, : start_pos + seq_len]
        # values = self.cache_v[:batch_size, : start_pos + seq_len]
        # print(f"Cache shapes - K: {keys.shape}, V: {values.shape}")

        # Update KV cache
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        # Get keys and values from cache
        keys = self.cache_k[:batch_size, : start_pos + seq_len].to(device=x.device)
        values = self.cache_v[:batch_size, : start_pos + seq_len].to(device=x.device)
        print(f"Cache shapes - K: {keys.shape}, V: {values.shape}")


        # Repeat KV heads
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # Transpose for attention computation
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        print(f"After transpose - Q: {xq.shape}, K: {keys.shape}, V: {values.shape}")

        # Compute attention scores
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        print(f"Attention scores shape: {scores.shape}")
        
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # Compute attention output
        output = torch.matmul(scores, values)
        print(f"After attention shape: {output.shape}")
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        print(f"Before final projection: {output.shape}")
        
        output = self.wo(output)
        print(f"Final output shape: {output.shape}")
        
        return output