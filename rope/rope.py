import torch

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # As written in the paragraph 3.2.2 of the paper
    # >> In order to generalize our results in 2D to any xi ∈ Rd where **d is even**, [...]
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    
    print(f"\nPrecomputing Rotary Embeddings:")
    print(f"Input dimensions - head_dim: {head_dim}, seq_len: {seq_len}")
    
    # Build the theta parameter
    # According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    theta_numerator = torch.arange(0, head_dim, 2).float()
    print(f"Theta numerator shape: {theta_numerator.shape}")
    
    # Shape: (Head_Dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    print(f"Theta shape: {theta.shape}")
    
    # Construct the positions (the "m" parameter)
    # Shape: (Seq_Len)
    m = torch.arange(seq_len, device=device)
    print(f"Position indices shape: {m.shape}")
    
    # Multiply each theta by each position using the outer product
    # Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs = torch.outer(m, theta).float()
    print(f"Frequencies shape after outer product: {freqs.shape}")
    
    # Compute complex numbers in the polar form c = R * exp(m * theta), where R = 1
    # (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    print(f"Complex frequencies shape: {freqs_complex.shape}")
    
    return freqs_complex

def apply_rotary_embeddings(x, freqs_complex, device=None):
    # Print input shapes for debugging
    print(f"Input tensor shape: {x.shape}")
    print(f"Frequencies tensor shape: {freqs_complex.shape}")
    
    # Split into real and imaginary components
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    print(f"Input as complex numbers shape: {x_complex.shape}")
    
    # Only take the frequencies we need for our sequence length
    seq_len = x.shape[1]
    freqs_complex = freqs_complex[:seq_len]
    
    # Expand frequencies to match batch and heads dimensions
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    print(f"Expanded frequencies shape: {freqs_complex.shape}")
    
    # Perform the rotation in complex space
    x_rotated = x_complex * freqs_complex
    x_rotated = torch.view_as_real(x_rotated).flatten(-2)
    
    # Cast back to the input dtype
    x_rotated = x_rotated.type_as(x)
    
    return x_rotated