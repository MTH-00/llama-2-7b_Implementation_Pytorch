# LLaMA-2-7B Implementation in PyTorch

A PyTorch-based implementation of the **LLaMA-2 (7B)** model architecture, modularly structured for clarity and ease of experimentation.

---

## Repository Structure

- **attention/** – Self-attention layer implementation.
- **embeding/** – Embedding modules (including token and positional embeddings).
- **encoder/** – Encoder block definitions for the model.
- **ffn/** – Feed-forward network modules.
- **norm/** – Normalization layers (e.g., RMSNorm).
- **rope/** – Rotary Position Embeddings (RoPE) implementation.
- **tokenizer/** – Tokenizer utilities and related files.
- **transformer/** – Main transformer architecture/orchestration.
- **example_usage.py** – Demonstration script showing how to load and use the model.
- **requirements.txt** – Python dependencies required for running the code.

---

## Highlights

- **Modular Design**: Architecture is broken into distinct components—attention, embeddings, encoder, FFN, normalization, RoPE, transformer orchestration—facilitating readability and experimentation.
- **RoPE Support**: Includes rotary positional embeddings for efficient handling of long-range token dependencies.
- **Easy to Use**: `example_usage.py` offers a clear, practical entry point for usage and integration.

---

## Suggested Workflow

1. **Explore the modules**: Dive into folders like `attention/`, `rope/`, `encoder/`, etc., to understand each model component.
2. **Run the example**: Use `example_usage.py` to load the model and run inference, serving as a starting point for tweaks.
3. **Customize or build**: Modify components or combine them into custom pipelines for experimentation or extended use cases.
4. **Integrate into projects**: Incorporate this modular implementation into broader training or fine-tuning workflows.

---

## Background (LLaMA-2 Context)

LLaMA-2 is Meta’s family of large language models, with versions ranging from **7B to 70B parameters** :contentReference[oaicite:0]{index=0}. This lightweight implementation focuses on the **7B variant**, a decoder-only transformer featuring techniques such as **Grouped-Query Attention**, **Rotary Positional Embeddings (RoPE)**, and **Root Mean Square Normalization (RMSNorm)** :contentReference[oaicite:1]{index=1}.

---

## Quick Start (Suggested)

```bash
git clone https://github.com/Shameerisb/llama-2-7b_Implementation_Pytorch
cd llama-2-7b_Implementation_Pytorch
pip install -r requirements.txt
python example_usage.py
