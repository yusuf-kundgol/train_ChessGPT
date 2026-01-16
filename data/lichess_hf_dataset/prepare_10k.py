"""Prepare a 10,000 sample subset of the chess dataset with '+' removed."""

import os
import shutil
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import pickle

num_proc = 8
dtype = np.uint8

if __name__ == "__main__":
    dataset_path = "adamkarvonen/chess_games"
    file_path = "lichess_6gb_blocks.zip"

    # Output directory for the 10k dataset
    output_dir = os.path.join(os.path.dirname(__file__), "..", "lichess_10k_no_plus")
    os.makedirs(output_dir, exist_ok=True)

    # Copy meta.pkl to the new directory
    src_meta = os.path.join(os.path.dirname(__file__), "meta.pkl")
    dst_meta = os.path.join(output_dir, "meta.pkl")
    shutil.copy(src_meta, dst_meta)
    print(f"Copied meta.pkl to {output_dir}")

    print("Loading dataset from HuggingFace...")
    dataset = load_dataset(dataset_path, data_files=file_path)

    # Take only first 10,000 samples
    num_samples = 10000
    print(f"Selecting {num_samples} samples...")
    dataset = dataset["train"].select(range(num_samples))

    # Split into train/val (99% train, 1% val)
    split_dataset = dataset.train_test_split(test_size=0.01, seed=2357, shuffle=True)
    split_dataset["val"] = split_dataset.pop("test")

    print(f"Train samples: {len(split_dataset['train'])}")
    print(f"Val samples: {len(split_dataset['val'])}")

    # Load tokenizer metadata
    with open(src_meta, "rb") as f:
        meta = pickle.load(f)

    stoi = meta["stoi"]
    column_name = "transcript"

    def process(example):
        # Remove '+' characters from the transcript
        text = example[column_name].replace('+', '')
        ids = np.array([stoi[c] for c in text], dtype=dtype)
        return {"ids": ids, "len": len(ids)}

    # Tokenize
    tokenized = split_dataset.map(
        process,
        remove_columns=[column_name],
        desc="tokenizing",
        num_proc=num_proc,
    )

    # Write to train.bin and val.bin in the new directory
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        print(f"{split} has {arr_len:,} tokens")
        filename = os.path.join(output_dir, f"{split}.bin")
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))

        idx = 0
        for i in tqdm(range(len(dset)), desc=f"writing {filename}"):
            arr[idx : idx + dset[i]["len"]] = dset[i]["ids"]
            idx += dset[i]["len"]
        arr.flush()

    print(f"\nDone! Created dataset in: {output_dir}")
    print("To train, run:")
    print("  python train.py --dataset=lichess_10k_no_plus --wandb_log=False")
