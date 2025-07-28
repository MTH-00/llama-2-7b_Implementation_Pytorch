import os
from TOKENIZER_with_dims import Tokenizer, TOKENIZER_MODEL

# Update the TOKENIZER_MODEL path
TOKENIZER_MODEL = os.path.join(os.path.dirname(__file__), 'tokenizer', 'tokenizer.model')

# Initialize tokenizer
tokenizer = Tokenizer()

# Test text
text = "Hello, this is a test message!"

# Encode with both BOS and EOS tokens
encoded = tokenizer.encode(text, bos=True, eos=True)

# Decode back
decoded = tokenizer.decode(encoded)