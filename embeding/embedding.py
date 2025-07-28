from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from typing import Optional
from attention.attention import ModelArgs

@dataclass
class EmbeddingConfig:
    dim: int = 4096  # Embedding dimension
    vocab_size: int = -1  # Will be set from tokenizer
    max_seq_len: int = 2048
    dropout: float = 0.1
    device: str = None
    vocab_size: int = -1
    device: Optional[str] = None    

class LlamaEmbedding(nn.Module):
    def __init__(self, config: EmbeddingConfig):
        super().__init__()
        self.config = config
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        print(f"Initialized embedding layer with shape: {self.token_embedding.weight.shape}")
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize embedding weights similar to Llama's approach
        nn.init.normal_(self.token_embedding.weight, mean=0.0, 
                       std=math.sqrt(2.0 / (5 * self.config.dim)))
        print(f"Reset parameters with std: {math.sqrt(2.0 / (5 * self.config.dim))}")
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Print input dimensions
        print(f"\nInput token_ids shape: {token_ids.shape}")
        
        # Get embeddings from the embedding layer
        embeddings = self.token_embedding(token_ids)
        print(f"After embedding shape: {embeddings.shape}")
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        print(f"After dropout shape: {embeddings.shape}")
        
        return embeddings

def create_embedding_layer(tokenizer_vocab_size: int, 
                         embedding_dim: int = 4096,
                         max_seq_len: int = 2048,
                         dropout: float = 0.1,
                         device: str = None) -> LlamaEmbedding:
    """Helper function to create an embedding layer with the given parameters."""
    config = EmbeddingConfig(
        dim=embedding_dim,
        vocab_size=tokenizer_vocab_size,
        max_seq_len=max_seq_len,
        dropout=dropout,
        device=device
    )
    return LlamaEmbedding(config).to(device if device else 'cpu')