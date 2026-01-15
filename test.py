import pickle
import os

# Load the meta.pkl file
meta_path = 'data/chess_1000_no_+/meta.pkl'

if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    print(f"Vocabulary size: {meta['vocab_size']}")
    print("\nTokenizer Dictionary (stoi - string to index):")
    print("=" * 50)
    
    if 'stoi' in meta:
        stoi = meta['stoi']
        # Sort by token index to show all 32 tokens in order
        sorted_tokens = sorted(stoi.items(), key=lambda x: x[1])
        
        for token, idx in sorted_tokens:
            print(f"Token {idx:2d}: '{token}'")
    
    print("\n" + "=" * 50)
    print("\nTokenizer Dictionary (itos - index to string):")
    print("=" * 50)
    
    if 'itos' in meta:
        itos = meta['itos']
        for idx, token in enumerate(itos):
            print(f"Index {idx:2d}: '{token}'")
    
    print("\n" + "=" * 50)
    print("\nComplete meta.pkl contents:")
    print(meta)
else:
    print(f"Meta file not found at {meta_path}")
    print("\nBased on the dataset name 'chess_1000_no_+', this is likely a chess move dataset.")
    print("For chess notation, the typical 32-token vocabulary would be:")
    print("\nChess Tokenizer (32 tokens):")
    print("=" * 50)
    
    # Standard chess tokens for algebraic notation without '+'
    tokens = {
        # Files (columns)
        'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7,
        # Ranks (rows)
        '1': 8, '2': 9, '3': 10, '4': 11, '5': 12, '6': 13, '7': 14, '8': 15,
        # Pieces
        'N': 16,  # Knight
        'B': 17,  # Bishop
        'R': 18,  # Rook
        'Q': 19,  # Queen
        'K': 20,  # King
        # Special notation
        'x': 21,  # Capture
        '#': 22,  # Checkmate
        'O': 23,  # Castling (O-O or O-O-O)
        '-': 24,  # Castling separator or move separator
        '=': 25,  # Promotion
        # Newline/separator
        '\n': 26,
        # Additional possible tokens
        ' ': 27,  # Space
        '.': 28,  # Move number separator (e.g., "1.e4")
        '/': 29,  # Alternative separator
        '_': 30,  # Padding or special token
        '<': 31,  # End of sequence or special token
    }
    
    print("\nEstimated token mapping (stoi):")
    for token, idx in sorted(tokens.items(), key=lambda x: x[1]):
        print(f"Token {idx:2d}: '{token}'")