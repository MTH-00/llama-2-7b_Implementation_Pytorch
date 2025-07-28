import torch
import torch.nn as nn
from encoder.encoder import EncoderBlock
from norm.norm import RMSNorm
from embeding.embedding import create_embedding_layer
from rope.rope import precompute_theta_pos_frequencies
from attention.attention import ModelArgs  # Same as TransformerConfig
from tokenizer.TOKENIZER_with_dims import Tokenizer


class Transformer(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.vocab_size = tokenizer.n_words

        # Embedding layer
        self.tok_embeddings = create_embedding_layer(
            tokenizer_vocab_size=tokenizer.n_words,
            embedding_dim=args.dim,
            max_seq_len=args.max_seq_len,
            dropout=0.1,  # Optional, use args if configurable
            device=args.device or 'cpu'
        )

        # Encoder layers
        self.layers = nn.ModuleList([
            EncoderBlock(args) for _ in range(args.n_layers)
        ])

        # Final RMSNorm
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        # Output head
        self.output = nn.Linear(args.dim, tokenizer.n_words, bias=False)

        # Rotary position encodings
        self.freqs_complex = precompute_theta_pos_frequencies(
            head_dim=args.dim // args.n_heads,
            seq_len=args.max_seq_len * 2,
            device=args.device or 'cpu'
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # tokens: [B, T]
        x = self.tok_embeddings(tokens)
        freqs_complex = self.freqs_complex[start_pos : start_pos + x.shape[1]]

        for layer in self.layers:
            x = layer(x, start_pos, freqs_complex)

        x = self.norm(x)
        logits = self.output(x)
        return logits


def create_transformer_model(tokenizer_vocab_size: int, 
                              dim: int = 4096,
                              max_seq_len: int = 2048,
                              dropout: float = 0.1,
                              device: str = None) -> Transformer:
    """Creates a full Transformer model with standard settings."""
    
    config = ModelArgs(
        dim=dim,
        vocab_size=tokenizer_vocab_size,
        max_seq_len=max_seq_len,
        dropout=dropout,
        device=device,
        n_layers=12,
        n_heads=32,
        n_kv_heads=4,  # or less if using GQA
        max_batch_size=32,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=1e-5
    )
    
    tokenizer = Tokenizer()
    return Transformer(config, tokenizer).to(device if device else 'cpu')