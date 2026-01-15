import torch
import os

checkpoint_path = 'out-chess-8layer/ckpt.pt'

if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("\n" + "="*60)
    print("CHECKPOINT CONTENTS")
    print("="*60)
    
    print("\nKeys in checkpoint:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    
    model_args = checkpoint['model_args']
    for key, value in model_args.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("MODEL STATE DICT")
    print("="*60)
    
    state_dict = checkpoint['model']
    print(f"\nTotal parameters in state_dict: {len(state_dict)}")
    print("\nFirst 10 layer names:")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'scalar'
        print(f"  {key}: {shape}")
    
    print("\n" + "="*60)
    print("TRAINING STATE")
    print("="*60)
    print(f"  iter_num: {checkpoint['iter_num']}")
    print(f"  best_val_loss: {checkpoint['best_val_loss']}")
    
    print("\n" + "="*60)
    print("✓ Checkpoint is valid and ready for fine-tuning!")
    print("="*60)
    print("\nTo start fine-tuning, run:")
    print("  python train.py config/finetune_chess_8layer.py")
    
else:
    print(f"✗ Checkpoint not found at: {checkpoint_path}")
    print("\nPlease run: python download_8layer_checkpoint.py")
