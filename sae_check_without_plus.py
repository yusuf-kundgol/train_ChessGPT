#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import torch as t
import torch
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt

# Add paths for chess_utils
sys.path.insert(0, '/root/chessgpt_sae')
sys.path.insert(0, '/root/chessgpt_git/chessgpt_git/SAE_BoardGameEval')
sys.path.insert(0, '/root/chessgpt_git/chessgpt_git/SAE_BoardGameEval/circuits')

import chess_utils

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

with open('/root/train_ChessGPT/data/lichess_hf_dataset/meta.pkl', 'rb') as picklefile:
    meta = pickle.load(picklefile)
print(f"Loaded meta.pkl with {len(meta['itos'])} tokens")


# In[ ]:


import torch
import sys
sys.path.insert(0, '/root/train_ChessGPT')
from model import GPTConfig, GPT

# Load the fine-tuned model from nanoGPT checkpoint
ckpt_path = '/root/train_ChessGPT/out-chess-8layer/ckpt.pt'
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

print(f"Loaded fine-tuned model from {ckpt_path}")
print(f"Model config: {checkpoint['model_args']}")


# In[ ]:


import circuits.eval_sae_as_classifier as eval_sae

train_dataset_name = "/root/train_ChessGPT/chess_train_dataset_check_10k.pkl"

if os.path.exists(train_dataset_name):
    print("Loading cached dataset...")
    with open(train_dataset_name, "rb") as f:
        train_data = pickle.load(f)
else:
    print("Creating dataset (this may take a few minutes)...")
    train_data = eval_sae.construct_dataset(
        False,
        [chess_utils.board_to_check_state],
        10000,
        split="train",
        device=device,
        precompute_dataset=True,
    )
    with open(train_dataset_name, "wb") as f:
        pickle.dump(train_data, f)
    print(f"Saved dataset to {train_dataset_name}")

print(f"Dataset loaded with {len(train_data['encoded_inputs'])} games")


# In[ ]:


# Count positions with check ('+' symbol)
games_close_to_check = []

for i, pgn in enumerate(train_data['decoded_inputs']):
    for j in range(len(pgn)):
        if pgn[j] == '+':
            games_close_to_check.append((i, j))

print(f"Found {len(games_close_to_check)} positions with check")
print(f"Check positions available for evaluation: {train_data['board_to_check_state'].nonzero().shape[0]}")


# In[ ]:


# Ready to evaluate - model and data loaded
print("Setup complete! Ready to run evaluation.")


# In[ ]:


t.set_grad_enabled(False)
from copy import copy
import chess

illegal_moves=[]
move_results_with_plus=[]
move_results_without_plus=[]

def find_last_ocurrence(arr, value):
    for i in range(-1, -len(arr), -1):
        if arr[i]==value:
            return i
    raise Exception

def find_next_space(string, start_index):
    for i in range(start_index, len(string)):
        if string[i] == ' ':
            return i
    return len(string)

def is_legal_move(board, move_string):
    try:
        move = board.parse_san(move_string)
        if move in board.legal_moves:
            return "legal"
        else:
            return "not in legal moves"
    except (chess.IllegalMoveError, chess.AmbiguousMoveError) as e:
        illegal_moves.append(move_string)
        if isinstance(e, chess.IllegalMoveError):
            return "illegal"
        else:
            return "ambiguous"
    except Exception as e:
        return "illegal"

num_samples = 10000
num_sampled = 0
same_moves_predicted = 0
num_available = train_data['board_to_check_state'].nonzero().shape[0]
current_sample = 0

print(f"Starting evaluation with {num_available} available samples...")

while num_sampled < num_samples and current_sample < num_available:
    pgn_idx, move_idx, _, _, _ = train_data['board_to_check_state'].nonzero()[current_sample]
    current_sample += 1
    if move_idx == 0 or not 4 == train_data['encoded_inputs'][pgn_idx][move_idx-1]:
        continue
    num_sampled += 1

    if num_sampled % 100 == 0:
        print(f"Processed {num_sampled} samples...")

    encoded_input = train_data['encoded_inputs'][pgn_idx][:move_idx]
    current_board_string = "".join([meta["itos"][c] for c in encoded_input])
    current_board = chess_utils.pgn_string_to_board(current_board_string)
    current_board_last_index = len(current_board_string) - 1

    encoded_input_without_plus = copy(train_data['encoded_inputs'][pgn_idx][:move_idx])
    del encoded_input_without_plus[find_last_ocurrence(encoded_input_without_plus, 2)]
    assert len(encoded_input) - 1 == len(encoded_input_without_plus)

    token_move_with_plus = model.generate(t.tensor(encoded_input).to(device).to(torch.int64).unsqueeze(dim=0), 7)
    token_move_without_plus = model.generate(t.tensor(encoded_input_without_plus).to(device).to(torch.int64).unsqueeze(dim=0), 7)

    string_move_with_plus = "".join([meta["itos"][c] for c in token_move_with_plus.squeeze().tolist()])
    string_move_without_plus = "".join([meta["itos"][c] for c in token_move_without_plus.squeeze().tolist()])

    with_plus_index = find_next_space(string_move_with_plus, current_board_last_index)
    without_plus_index = find_next_space(string_move_without_plus, current_board_last_index - 1)

    move_with_plus = string_move_with_plus[current_board_last_index + 1:with_plus_index]
    move_without_plus = string_move_without_plus[current_board_last_index:without_plus_index]

    if move_with_plus == move_without_plus:
        same_moves_predicted += 1

    move_results_with_plus.append(is_legal_move(current_board, move_with_plus))
    move_results_without_plus.append(is_legal_move(current_board, move_without_plus))

print(f"\nDone! Sampled {num_sampled} positions")
print(f"Same moves predicted: {same_moves_predicted} ({100*same_moves_predicted/num_sampled:.1f}%)")


# In[35]:


import matplotlib.pyplot as plt

# Count the occurrences of "legal", "illegal", and "ambiguous" moves
with_plus_legal = move_results_with_plus.count("legal")
with_plus_illegal = move_results_with_plus.count("illegal")
with_plus_ambiguous = move_results_with_plus.count("ambiguous")

without_plus_legal = move_results_without_plus.count("legal")
without_plus_illegal = move_results_without_plus.count("illegal")
without_plus_ambiguous = move_results_without_plus.count("ambiguous")

# Create pie charts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Pie chart for moves with plus
sizes_with_plus = [with_plus_legal, with_plus_illegal, with_plus_ambiguous]
labels_with_plus = ['Legal', 'Illegal', 'Ambiguous']
colors_with_plus = ['#ff9999', '#66b3ff', '#99ff99']

ax1.pie(sizes_with_plus, labels=labels_with_plus, colors=colors_with_plus, autopct='%1.1f%%', startangle=90)
ax1.set_title('Moves with Plus')

# Add a legend for the "with plus" chart
ax1.legend(labels_with_plus, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

# Pie chart for moves without plus
ax2.pie([without_plus_legal, without_plus_illegal, without_plus_ambiguous], 
        labels=['Legal', 'Illegal', 'Ambiguous'], 
        autopct='%1.1f%%')
ax2.set_title('Moves without Plus')

plt.tight_layout()
plt.show()
#print(move_with_plus, move_without_plus)


# In[ ]:


# Print summary statistics
print("=== RESULTS SUMMARY ===\n")

with_plus_legal = move_results_with_plus.count("legal")
with_plus_illegal = move_results_with_plus.count("illegal")
with_plus_ambiguous = move_results_with_plus.count("ambiguous")
with_plus_not_in_legal = move_results_with_plus.count("not in legal moves")

without_plus_legal = move_results_without_plus.count("legal")
without_plus_illegal = move_results_without_plus.count("illegal")
without_plus_ambiguous = move_results_without_plus.count("ambiguous")
without_plus_not_in_legal = move_results_without_plus.count("not in legal moves")

total = len(move_results_with_plus)

print(f"WITH '+' in context:")
print(f"  Legal: {with_plus_legal} ({100*with_plus_legal/total:.1f}%)")
print(f"  Illegal: {with_plus_illegal} ({100*with_plus_illegal/total:.1f}%)")
print(f"  Ambiguous: {with_plus_ambiguous} ({100*with_plus_ambiguous/total:.1f}%)")
print(f"  Not in legal moves: {with_plus_not_in_legal} ({100*with_plus_not_in_legal/total:.1f}%)")

print(f"\nWITHOUT '+' in context:")
print(f"  Legal: {without_plus_legal} ({100*without_plus_legal/total:.1f}%)")
print(f"  Illegal: {without_plus_illegal} ({100*without_plus_illegal/total:.1f}%)")
print(f"  Ambiguous: {without_plus_ambiguous} ({100*without_plus_ambiguous/total:.1f}%)")
print(f"  Not in legal moves: {without_plus_not_in_legal} ({100*without_plus_not_in_legal/total:.1f}%)")

print(f"\nDifference in legal move %: {100*(with_plus_legal - without_plus_legal)/total:.1f}%")

