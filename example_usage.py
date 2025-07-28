import torch
from tokenizer.TOKENIZER_with_dims import Tokenizer
from embeding.embedding import create_embedding_layer
from rope.rope import precompute_theta_pos_frequencies
from attention.attention import SelfAttention, ModelArgs
from norm.norm import RMSNorm
from ffn.ffn import FeedForward, ModelArgs as FFNArgs
from encoder.encoder import EncoderBlock, ModelArgs as EncoderArgs

# Initialize tokenizer and model dimensions
tokenizer = Tokenizer()
embedding_dim = 4096
max_seq_len = 2048
num_heads = 32
num_kv_heads = 4  # For grouped-query attention
# # max_batch_size = 16

args = ModelArgs(
    dim=4096,
    n_heads=32,
    n_kv_heads=4,
    max_batch_size=32,
    max_seq_len=2048,
    multiple_of=256,
    ffn_dim_multiplier=None,
    norm_eps=1e-5
)



# Create embedding layer
embedding_layer = create_embedding_layer(
    tokenizer_vocab_size=tokenizer.n_words,
    embedding_dim=embedding_dim,
    max_seq_len=max_seq_len
)

# #Initialize attention configuration
# attention_args = ModelArgs(
#     dim=embedding_dim,
#     n_heads=num_heads,
#     n_kv_heads=num_kv_heads,
#     max_seq_len=max_seq_len
# )

# Create attention layer
# attention_layer = SelfAttention(attention_args)

attention_layer = SelfAttention(args)

# Precompute RoPE frequencies
head_dim = embedding_dim // num_heads
freqs_complex = precompute_theta_pos_frequencies(
    head_dim=head_dim,
    seq_len=max_seq_len,
    device="cpu",
    theta=10000.0
)

# Example usage
text = "Hello, this is a test sentence."

# Tokenize the text (this will print token dimensions)
tokens = tokenizer.encode(text, bos=True, eos=True)
print(f"\nTokenized sequence length: {len(tokens)}")

# Convert tokens to tensor and add batch dimension
token_tensor = torch.tensor([tokens], dtype=torch.long)  # Shape: [1, seq_len]
print(f"Token tensor shape: {token_tensor.shape}")

# Get embeddings (this will print embedding dimensions)
embeddings = embedding_layer(token_tensor)
print(f"Embeddings shape: {embeddings.shape}")

norm_layer = RMSNorm(embedding_dim)
normalized_embeddings = norm_layer(embeddings)


# Apply self-attention with RoPE
start_pos = 0  # For the first position in sequence
attention_output = attention_layer(
    x=normalized_embeddings,
    start_pos=start_pos,
    freqs_complex=freqs_complex
)
print(f"\nFinal attention output shape: {attention_output.shape}")

# print(f"\nToken 1 embedding: {embeddings[0, 0, :]}")
print("\nToken 1 embedding:", embeddings[0, 0, :].detach().numpy())

print(f"\nToken 1 normalized embedding: {normalized_embeddings[0, 0, :]}")


# # Initialize FFN
# ffn_args = FFNArgs(
#     dim=4096,
#     multiple_of=256,
#     ffn_dim_multiplier=None
# )

# ffn_layer = FeedForward(ffn_args)
ffn_layer  = FeedForward(args)

# Test FFN with the output from attention layer
ffn_output = ffn_layer(attention_output)

print(f"\nFFN output shape: {ffn_output.shape}")


# # Initialize Encoder Block
# class ModelArgs:
#     def __init__(self):
#         self.dim = 4096
#         self.n_heads = 32
#         self.n_kv_heads = 4
#         self.norm_eps = 1e-5
#         # Add missing cache-related parameters
#         self.max_batch_size = 32
#         self.max_seq_len = 2048

# encoder_args = ModelArgs()
# multiple_of=256,
# norm_eps=1e-5

encoder_block = EncoderBlock(args)

# Test encoder block with the previous output
encoder_output = encoder_block(attention_output, start_pos=0, freqs_complex=freqs_complex)

print(f"\nEncoder output shape: {encoder_output.shape}")




# import torch
# from tokenizer.TOKENIZER_with_dims import Tokenizer
# from transformer.transformer import Transformer
# from attention.attention import ModelArgs

# # Load tokenizer
# tokenizer = Tokenizer()

# # Define model args
# args = ModelArgs(
#     dim=4096,
#     n_heads=32,
#     n_kv_heads=4,
#     max_batch_size=32,
#     max_seq_len=2048,
#     multiple_of=256,
#     ffn_dim_multiplier=None,
#     norm_eps=1e-5,
#     vocab_size=tokenizer.n_words,
#     device=1
# )

# # Instantiate full model
# model = Transformer(args, tokenizer)

# # Input text
# text = "LLaMA transformers are efficient."
# tokens = tokenizer.encode(text, bos=True, eos=True)
# input_tensor = torch.tensor([tokens], dtype=torch.long)  # [1, seq_len]

# # Forward pass
# start_pos = 0
# with torch.no_grad():
#     logits = model(input_tensor, start_pos)

# print("Logits shape:", logits.shape)  # Should be [1, seq_len, vocab_size]
