from huggingface_hub import hf_hub_download
import os

# Create directory for the checkpoint
os.makedirs("out-chess", exist_ok=True)

# Download the checkpoint
print("Downloading checkpoint from Hugging Face...")
checkpoint_path = hf_hub_download(
    repo_id="adamkarvonen/chess_llms",
    filename="lichess_8layers_ckpt_no_optimizer.pt",
    local_dir="out-chess"
)

print(f"Checkpoint downloaded to: {checkpoint_path}")
print("\nRenaming to ckpt.pt for training...")
os.rename(checkpoint_path, "out-chess/ckpt.pt")
print("Done! Checkpoint ready at: out-chess/ckpt.pt")
