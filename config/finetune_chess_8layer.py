import time

# Output directory for fine-tuned model
out_dir = 'out-chess-8layer'

# Start from the downloaded 8-layer checkpoint
init_from = 'resume'

# Evaluation settings
eval_interval = 500
eval_iters = 100
log_interval = 10

# Evaluation only mode - set to True if you just want to evaluate the checkpoint
eval_only = False

# Logging
wandb_log = True
wandb_project = 'chess-gpt'
wandb_run_name = 'finetune-8layer-chess-1000-' + str(int(time.time()))

# Dataset - chess moves without '+' symbol
dataset = 'chess_1000_no_+'

# Training settings
batch_size = 64
gradient_accumulation_steps = 1

# The checkpoint has block_size = 1023, which will be loaded automatically
# We don't need to specify n_layer, n_head, n_embd - they'll be loaded from checkpoint
# The 8-layer model has: n_layer=8, n_head=8, n_embd=512

# Fine-tuning hyperparameters (lower learning rate than training from scratch)
learning_rate = 1e-4  # Conservative learning rate for fine-tuning
max_iters = 10000     # Adjust based on your dataset size
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate decay settings
decay_lr = True
warmup_iters = 100
lr_decay_iters = 10000  # Should match max_iters
min_lr = 1e-5           # Minimum learning rate

# Dropout for fine-tuning (can add some regularization)
dropout = 0.0  # Start with 0.0, increase to 0.1 if overfitting

# Save checkpoints
always_save_checkpoint = True

# System
device = 'cuda'
dtype = 'bfloat16'  # Will auto-detect GPU capabilities
compile = True  # Use PyTorch 2.0 compilation for speed
