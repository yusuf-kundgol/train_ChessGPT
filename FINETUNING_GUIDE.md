# Fine-tuning the 8-Layer Chess GPT Model

## ✓ Setup Complete!

The 8-layer Chess GPT model from HuggingFace has been downloaded and is ready for fine-tuning.

### Model Details
- **Source**: `adamkarvonen/8LayerChessGPT2` (HuggingFace)
- **Architecture**: 8 layers, 8 heads, 512 embedding dimension
- **Vocabulary**: 32 tokens (chess notation)
- **Context length**: 1023 tokens
- **Location**: `out-chess-8layer/ckpt.pt`

### Files Created

1. **`download_8layer_checkpoint.py`** - Script to download and convert the model from HuggingFace
2. **`config/finetune_chess_8layer.py`** - Training configuration for fine-tuning
3. **`verify_checkpoint.py`** - Script to verify the checkpoint is valid

### How to Start Fine-tuning

#### Option 1: Basic fine-tuning (recommended)
```bash
python train.py config/finetune_chess_8layer.py
```

#### Option 2: With custom settings
```bash
python train.py config/finetune_chess_8layer.py \
    --learning_rate=5e-5 \
    --max_iters=5000 \
    --batch_size=32
```

#### Option 3: Without wandb logging
```bash
python train.py config/finetune_chess_8layer.py --wandb_log=False
```

### Configuration Highlights

The fine-tuning config (`config/finetune_chess_8layer.py`) includes:

- **Learning rate**: 1e-4 (conservative for fine-tuning)
- **Max iterations**: 10,000
- **Batch size**: 64
- **Dataset**: `chess_1000_no_+` (your chess moves dataset)
- **Evaluation**: Every 500 steps
- **Checkpointing**: Saves when validation loss improves

### Adjusting Hyperparameters

Edit `config/finetune_chess_8layer.py` to customize:

```python
# Training duration
max_iters = 10000

# Learning rate (lower = more conservative)
learning_rate = 1e-4

# Batch size (larger = more stable but needs more memory)
batch_size = 64

# Regularization (0.0 = no dropout, 0.1 = light regularization)
dropout = 0.0
```

### Monitoring Training

If you have wandb enabled:
1. Training logs will appear in your wandb project: `chess-gpt`
2. Run name: `finetune-8layer-chess-1000-<timestamp>`

Without wandb:
- Watch the terminal output for loss values
- Checkpoints saved to: `out-chess-8layer/ckpt.pt`

### After Training

Your fine-tuned model will be saved in `out-chess-8layer/ckpt.pt`, which you can use for:
- Generating chess moves: `python sample.py --out_dir=out-chess-8layer`
- Continued training: Run the same command again (it will resume)
- Evaluation: `python train.py config/finetune_chess_8layer.py --eval_only=True`

### Troubleshooting

**If you need to re-download the model:**
```bash
rm -rf out-chess-8layer/
python download_8layer_checkpoint.py
```

**If training fails due to memory:**
- Reduce `batch_size` in the config file
- Reduce `block_size` if needed
- Set `compile=False` to reduce memory usage

**To verify your checkpoint:**
```bash
python verify_checkpoint.py
```
