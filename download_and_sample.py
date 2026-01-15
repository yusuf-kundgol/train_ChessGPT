from datasets import load_dataset
import pandas as pd
import os

# Download the dataset
print("Downloading dataset from HuggingFace...")
dataset_path = "adamkarvonen/chess_games"
file_path = "lichess_6gb_blocks.zip"
dataset = load_dataset(dataset_path, data_files=file_path)

# Convert to pandas DataFrame
print("Converting to DataFrame...")
df = pd.DataFrame(dataset["train"])

# Sample 1000 entries
print("Sampling 1000 entries...")
df_sampled = df.sample(n=1000, random_state=42)

# Remove all occurrences of "+" from the transcript column
print("Removing '+' characters...")
df_sampled['transcript'] = df_sampled['transcript'].str.replace('+', '', regex=False)

# Save to CSV
output_path = "data/chess_1000_no_x/chess_1000_no_x.csv"
os.makedirs("data/chess_1000_no_x", exist_ok=True)
print(f"Saving to {output_path}...")
df_sampled.to_csv(output_path, index=False)

print(f"Done! Saved {len(df_sampled)} entries to {output_path}")
print(f"Columns in dataset: {df_sampled.columns.tolist()}")
print(f"\nFirst row preview:")
print(df_sampled.iloc[0])
