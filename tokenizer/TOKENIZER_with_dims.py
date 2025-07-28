# Taken from llama code and lightly modified
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import struct
from typing import List

from sentencepiece import SentencePieceProcessor

# Update this line
TOKENIZER_MODEL = os.path.join(os.path.dirname(__file__), "tokenizer.model")
TOKENIZER_BIN = "tokenizer.bin" # binary version of the tokenizer for inference in C

class Tokenizer:
    def __init__(self):
        model_path = TOKENIZER_MODEL
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        print(f"Loaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        print(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        print(f"\nInput string length: {len(s)} characters")
        print(f"Input text: {s}")
        
        t = self.sp_model.encode(s)
        print(f"\nAfter initial encoding: {len(t)} tokens")
        print("Token mapping:")
        for token_id in t:
            piece = self.sp_model.id_to_piece(token_id)
            print(f"Token {token_id}: {piece}")
        
        if bos:
            t = [self.bos_id] + t
            print(f"\nAfter adding BOS token: {len(t)} tokens")
            print(f"Added BOS token {self.bos_id}: {self.sp_model.id_to_piece(self.bos_id)}")
            
        if eos:
            t = t + [self.eos_id]
            print(f"\nAfter adding EOS token: {len(t)} tokens")
            print(f"Added EOS token {self.eos_id}: {self.sp_model.id_to_piece(self.eos_id)}")
            
        print(f"\nFinal output dimension: {len(t)} tokens")
        print("Final token sequence:")
        for token_id in t:
            piece = self.sp_model.id_to_piece(token_id)
            print(f"Token {token_id}: {piece}")
        return t

    def decode(self, t: List[int]) -> str:
        print(f"\nDecoding input dimension: {len(t)} tokens")
        print("Input tokens:")
        for token_id in t:
            piece = self.sp_model.id_to_piece(token_id)
            print(f"Token {token_id}: {piece}")
            
        result = self.sp_model.decode(t)
        print(f"Decoded output length: {len(result)} characters")
        print(f"Decoded text: {result}")
        return result

    def export(self):
        print("\nStarting export process...")
        tokens, scores = [], []
        for i in range(self.n_words):
            t = self.sp_model.id_to_piece(i)
            s = self.sp_model.get_score(i)
            if i == self.bos_id:
                t = '\n<s>\n'
            elif i == self.eos_id:
                t = '\n</s>\n'
            elif len(t) == 6 and t.startswith('<0x') and t.endswith('>'):
                t = chr(int(t[3:5], 16))
            t = t.replace(' ', ' ')
            b = t.encode('utf-8')

            tokens.append(b)
            scores.append(s)
            
            if i % 1000 == 0:
                print(f"Processed {i}/{self.n_words} tokens")
        
        max_token_length = max(len(t) for t in tokens)
        print(f"\nMaximum token length: {max_token_length} bytes")
        print(f"Total number of tokens: {len(tokens)}")

        with open(TOKENIZER_BIN, 'wb') as f:
            f.write(struct.pack("I", max_token_length))
            for bytes, score in zip(tokens, scores):
                f.write(struct.pack("fI", score, len(bytes)))
                f.write(bytes)
        print(f"\nExported binary tokenizer to {TOKENIZER_BIN}")

if __name__ == "__main__":
    t = Tokenizer()
    
    # Test encoding and decoding
    test_text = "Hi, this is a test message!"
    print(f"\nTesting with text: {test_text}")
    
    # Encode with both BOS and EOS tokens
    encoded = t.encode(test_text, bos=True, eos=True)
    print(f"\nEncoded token IDs: {encoded}")
    
    # Decode back
    decoded = t.decode(encoded)
    
    # Export the tokenizer
    t.export()