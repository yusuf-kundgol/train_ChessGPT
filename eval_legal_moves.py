"""Evaluate fine-tuned model's legal move accuracy with/without '+' in context."""
import os
import sys
import torch as t
import torch
import numpy as np
import pickle
from copy import copy
import chess
from tqdm import tqdm

sys.path.insert(0, '/root/chessgpt_sae')
sys.path.insert(0, '/root/chessgpt_git/chessgpt_git/SAE_BoardGameEval')
sys.path.insert(0, '/root/chessgpt_git/chessgpt_git/SAE_BoardGameEval/circuits')
sys.path.insert(0, '/root/train_ChessGPT')

import chess_utils
from model import GPTConfig, GPT

# Config
NUM_SAMPLES = 10000  # Evaluate on full dataset

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load meta
with open('/root/train_ChessGPT/data/lichess_hf_dataset/meta.pkl', 'rb') as f:
    meta = pickle.load(f)

# Load model
print('Loading model...')
ckpt_path = '/root/train_ChessGPT/out-chess-8layer/ckpt.pt'
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
for k, v in list(state_dict.items()):
    if k.startswith('_orig_mod.'):
        state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)
print(f'Model loaded: {checkpoint["model_args"]}')

# Load dataset
print('Loading dataset...')
with open('/root/train_ChessGPT/chess_train_dataset_check_10k.pkl', 'rb') as f:
    train_data = pickle.load(f)
print(f'Dataset loaded with {len(train_data["encoded_inputs"])} games')

# Helper functions
def find_last_ocurrence(arr, value):
    for i in range(-1, -len(arr), -1):
        if arr[i] == value:
            return i
    raise Exception("Value not found")

def find_next_space(string, start_index):
    for i in range(start_index, len(string)):
        if string[i] == ' ':
            return i
    return len(string)

def is_legal_move(board, move_string):
    try:
        move = board.parse_san(move_string)
        return 'legal' if move in board.legal_moves else 'not in legal moves'
    except:
        return 'illegal'

# Evaluation
t.set_grad_enabled(False)
move_results_with_plus = []
move_results_without_plus = []
num_sampled = 0
same_moves_predicted = 0
num_available = train_data['board_to_check_state'].nonzero().shape[0]
current_sample = 0

print(f'\nEvaluating {NUM_SAMPLES} samples from {num_available} available...\n')

pbar = tqdm(total=NUM_SAMPLES)
while num_sampled < NUM_SAMPLES and current_sample < num_available:
    pgn_idx, move_idx, _, _, _ = train_data['board_to_check_state'].nonzero()[current_sample]
    current_sample += 1
    if move_idx == 0 or train_data['encoded_inputs'][pgn_idx][move_idx-1] != 4:
        continue
    num_sampled += 1
    pbar.update(1)

    encoded_input = train_data['encoded_inputs'][pgn_idx][:move_idx]
    current_board_string = ''.join([meta['itos'][c] for c in encoded_input])
    current_board = chess_utils.pgn_string_to_board(current_board_string)
    current_board_last_index = len(current_board_string) - 1

    encoded_input_without_plus = copy(train_data['encoded_inputs'][pgn_idx][:move_idx])
    del encoded_input_without_plus[find_last_ocurrence(encoded_input_without_plus, 2)]

    token_move_with_plus = model.generate(t.tensor(encoded_input).to(device).long().unsqueeze(0), 7)
    token_move_without_plus = model.generate(t.tensor(encoded_input_without_plus).to(device).long().unsqueeze(0), 7)

    string_move_with_plus = ''.join([meta['itos'][c] for c in token_move_with_plus.squeeze().tolist()])
    string_move_without_plus = ''.join([meta['itos'][c] for c in token_move_without_plus.squeeze().tolist()])

    with_plus_index = find_next_space(string_move_with_plus, current_board_last_index)
    without_plus_index = find_next_space(string_move_without_plus, current_board_last_index - 1)

    move_with_plus = string_move_with_plus[current_board_last_index + 1:with_plus_index]
    move_without_plus = string_move_without_plus[current_board_last_index:without_plus_index]

    if move_with_plus == move_without_plus:
        same_moves_predicted += 1

    move_results_with_plus.append(is_legal_move(current_board, move_with_plus))
    move_results_without_plus.append(is_legal_move(current_board, move_without_plus))

pbar.close()

# Results
total = len(move_results_with_plus)
with_legal = move_results_with_plus.count('legal')
without_legal = move_results_without_plus.count('legal')

print(f'\n{"="*50}')
print(f'RESULTS SUMMARY ({total} samples)')
print(f'{"="*50}')
print(f'\nWITH "+" in context:')
print(f'  Legal moves: {with_legal}/{total} ({100*with_legal/total:.1f}%)')
print(f'\nWITHOUT "+" in context:')
print(f'  Legal moves: {without_legal}/{total} ({100*without_legal/total:.1f}%)')
print(f'\nSame move predicted: {same_moves_predicted}/{total} ({100*same_moves_predicted/total:.1f}%)')
print(f'Difference: {100*(with_legal - without_legal)/total:+.1f}%')
