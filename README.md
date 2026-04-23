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
python3 main.py
```

The script downloads `input.txt` if needed, trains the model, writes run
artifacts under `runs/<run_name>/`, and generates sample text at the end.

CUDA is used automatically if available:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

## Training artifacts

Each run stores:

- `runs/<run_name>/config.json` (full run config)
- `runs/<run_name>/checkpoints/latest.pt` (latest checkpoint)
- `runs/<run_name>/checkpoints/best.pt` (best validation checkpoint)
- `runs/<run_name>/logs/train.log` (console + file logs)
- `runs/<run_name>/metrics/metrics.csv` (structured metrics)
- `runs/<run_name>/samples/sample_step_*.txt` (eval snapshots)

## Resume training

```bash
python3 main.py --resume runs/<run_name>/checkpoints/latest.pt
```

You can override selected values while resuming (example):

```bash
python3 main.py \
  --resume runs/<run_name>/checkpoints/latest.pt \
  --max-iters 20000 \
  --run-name resumed_run
```

## Generate from a checkpoint

```bash
python3 generate.py \
  --checkpoint runs/<run_name>/checkpoints/best.pt \
  --prompt "ROMEO:" \
  --max-new-tokens 300
```

## Stop criteria and budget controls

Useful flags:

- `--early-stop-patience-evals`
- `--early-stop-min-delta`
- `--max-wall-time-minutes`
- `--hourly-cost-usd`
- `--budget-cap-usd`

Example:

```bash
python3 main.py \
  --hourly-cost-usd 0.30 \
  --budget-cap-usd 1.50 \
  --max-wall-time-minutes 240 \
  --early-stop-patience-evals 20 \
  --early-stop-min-delta 0.0005
```

## Vast.ai artifact sync

Run training inside `tmux`/`screen`, then sync artifacts regularly to avoid
losing progress when instances stop.

Example with `rsync`:

```bash
rsync -avh --progress runs/<run_name>/ /path/to/persistent-storage/<run_name>/
```

Before terminating an instance, do one final sync and verify both
`latest.pt` and `best.pt` exist in persistent storage.
