from huggingface_hub import hf_hub_download
from transformers import GPT2LMHeadModel
import torch
import os
import json

# Create directory for the 8-layer checkpoint
os.makedirs("out-chess-8layer", exist_ok=True)

# Download the model files from HuggingFace
print("Downloading 8-layer Chess GPT checkpoint from Hugging Face...")
print("Repository: adamkarvonen/8LayerChessGPT2")

try:
    # Download config and model files
    print("\nDownloading config.json...")
    config_path = hf_hub_download(
        repo_id="adamkarvonen/8LayerChessGPT2",
        filename="config.json",
        local_dir="out-chess-8layer",
        local_dir_use_symlinks=False
    )
    
    print("Downloading model.safetensors...")
    model_path = hf_hub_download(
        repo_id="adamkarvonen/8LayerChessGPT2",
        filename="model.safetensors",
        local_dir="out-chess-8layer",
        local_dir_use_symlinks=False
    )
    
    print("\n✓ Model files downloaded successfully!")
    print("Converting to ckpt.pt format compatible with train.py...")
    
    # Load the model using transformers
    model = GPT2LMHeadModel.from_pretrained("out-chess-8layer")
    
    # Read config to get model architecture
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    # Get state dict and transpose weights to match nanoGPT format
    state_dict = model.state_dict()
    
    # HuggingFace stores linear layer weights as (in_features, out_features)
    # but nanoGPT expects (out_features, in_features), so we need to transpose
    # Exception: lm_head and embeddings don't need transposing
    transposed_state_dict = {}
    for key, value in state_dict.items():
        if 'weight' in key and value.dim() == 2:
            # Check if this is a linear layer that needs transposing
            if any(x in key for x in ['c_attn', 'c_proj', 'c_fc']):
                transposed_state_dict[key] = value.t().contiguous()
            else:
                # Don't transpose lm_head or embeddings (wte, wpe)
                transposed_state_dict[key] = value
        else:
            transposed_state_dict[key] = value
    
    # Create checkpoint in the format expected by train.py
    checkpoint = {
        'model': transposed_state_dict,
        'model_args': {
            'n_layer': config_data['n_layer'],
            'n_head': config_data['n_head'],
            'n_embd': config_data['n_embd'],
            'block_size': config_data['n_positions'],
            'bias': True,  # GPT-2 uses bias
            'vocab_size': config_data['vocab_size'],
        },
        'iter_num': 0,
        'best_val_loss': float('inf'),
        'config': config_data
    }
    
    # Save as weights.pt
    torch.save(checkpoint, 'out-chess-8layer/weights.pt')
    
    print(f"\n✓ Conversion complete!")
    print(f"Location: out-chess-8layer/weights.pt")
    print(f"\nModel architecture:")
    print(f"  - Layers: {checkpoint['model_args']['n_layer']}")
    print(f"  - Heads: {checkpoint['model_args']['n_head']}")
    print(f"  - Embedding dimension: {checkpoint['model_args']['n_embd']}")
    print(f"  - Vocabulary size: {checkpoint['model_args']['vocab_size']}")
    print(f"  - Block size: {checkpoint['model_args']['block_size']}")
    print("\nYou can now use this checkpoint for fine-tuning by:")
    print("1. Using config: config/finetune_chess_8layer.py")
    print("2. Running: python train.py config/finetune_chess_8layer.py")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nPlease ensure:")
    print("1. Required packages are installed:")
    print("   pip install huggingface_hub transformers torch")
    print("2. You have internet connectivity")
    print("3. The repository exists and is accessible")
