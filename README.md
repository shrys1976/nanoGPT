# tinyShakespeare

A small character-level GPT-style language model built in PyTorch, inspired by Andrej Karpathy's nanoGPT series.

The model trains on Tiny Shakespeare dataset and learns to generate Shakespeare-like text one character at a time.

## Architecture

This is a decoder-only Transformer language model with:

- character-level tokenization
- token and positional embeddings
- causal multi-head self-attention
- Transformer blocks with residual connections
- pre-layer normalization
- feed-forward MLPs with GELU
- dropout
- final language modeling head

Current main settings:

```python
batch_size = 64
block_size = 256
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
```
This makes the total parameter count of the model about ~10.8M

The model predicts the next character using cross-entropy loss.

## Run

Install dependencies, then run:

```bash
python main.py
```

The script downloads `input.txt` if needed, trains the model, prints train/validation loss during training, and generates sample text at the end.

CUDA is used automatically if available:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```
