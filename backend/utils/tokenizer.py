"""
tokenizer.py — Tokenization helpers for NeuроScope.

WHAT IS TOKENIZATION?
  LLMs don't read words — they read "tokens".
  A token is roughly a word-piece. For example:
    "unbelievable" → ["un", "believ", "able"]   (3 tokens)
    "Hello world"  → ["Hello", " world"]         (2 tokens)
    "GPT"          → ["G", "PT"]                 (2 tokens)

  GPT-2 uses "Byte Pair Encoding" (BPE) — it merges common
  character sequences into single tokens during training.
  GPT-2's vocabulary has 50,257 unique tokens.

WHY THIS FILE EXISTS:
  The raw tokenizer output isn't user-friendly. This module:
  1. Cleans up token display (removes Ġ, Ċ artifacts)
  2. Maps token positions to character positions
  3. Identifies special tokens
  4. Provides token-level metadata for the frontend
"""

from typing import List, Dict, Any, Tuple
from model.loader import model_loader


# GPT-2 uses "Ġ" (unicode 0x0120) to mark tokens that start with a space
# "Ċ" marks newlines. We convert these to readable symbols.
SPACE_CHAR = "Ġ"
NEWLINE_CHAR = "Ċ"


def tokenize(text: str) -> Dict[str, Any]:
    """
    Full tokenization with metadata.

    Returns:
      {
        "tokens": ["The", " Eiffel", " Tower", ...],
        "token_ids": [464, 23455, 8514, ...],
        "num_tokens": 6,
        "clean_tokens": ["The", "Eiffel", "Tower", ...],   # no leading space
        "token_metadata": [{id, raw, clean, is_space, position}, ...]
        "char_to_token": [0, 0, 0, 1, 1, 1, 1, ...],      # char index → token index
        "token_to_chars": [(0,3), (3,10), ...]             # token index → (start, end) chars
      }
    """
    tokenizer = model_loader.get_tokenizer()

    # Main tokenization
    encoding = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
    token_ids = encoding["input_ids"][0].tolist()

    # Decode each token to get the raw string
    raw_tokens = [tokenizer.decode([tid]) for tid in token_ids]

    # Build metadata per token
    token_metadata = []
    clean_tokens = []

    for i, (tid, raw) in enumerate(zip(token_ids, raw_tokens)):
        # Check if token starts with a space (Ġ prefix in GPT-2)
        is_space_prefixed = raw.startswith(SPACE_CHAR) or raw.startswith(" ")
        is_newline = NEWLINE_CHAR in raw

        # Clean version for display
        clean = raw.replace(SPACE_CHAR, "").replace(NEWLINE_CHAR, "↵").strip()
        if not clean:
            clean = "[SPACE]" if is_space_prefixed else "[EMPTY]"

        clean_tokens.append(clean)

        token_metadata.append({
            "index": i,
            "token_id": tid,
            "raw": raw,
            "clean": clean,
            "display": f"·{clean}" if is_space_prefixed else clean,  # · shows space
            "is_space_prefixed": is_space_prefixed,
            "is_newline": is_newline,
            "is_special": tid in [
                tokenizer.eos_token_id,
                tokenizer.bos_token_id,
                tokenizer.pad_token_id,
            ],
            "byte_length": len(raw.encode("utf-8")),
        })

    # Character-to-token mapping
    # Useful for highlighting in the frontend: which token does char N belong to?
    char_to_token = []
    token_to_chars = []
    char_pos = 0

    for i, raw in enumerate(raw_tokens):
        display_raw = raw.replace(SPACE_CHAR, " ").replace(NEWLINE_CHAR, "\n")
        start = char_pos
        end = char_pos + len(display_raw)
        token_to_chars.append((start, end))

        for _ in range(len(display_raw)):
            char_to_token.append(i)

        char_pos = end

    return {
        "tokens": raw_tokens,
        "token_ids": token_ids,
        "num_tokens": len(token_ids),
        "clean_tokens": clean_tokens,
        "token_metadata": token_metadata,
        "char_to_token": char_to_token,
        "token_to_chars": token_to_chars,
    }


def decode_token_ids(token_ids: List[int]) -> str:
    """Convert a list of token IDs back to a string."""
    tokenizer = model_loader.get_tokenizer()
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def get_token_id(token_str: str) -> int:
    """Get the token ID for a specific string."""
    tokenizer = model_loader.get_tokenizer()
    ids = tokenizer.encode(token_str)
    return ids[0] if ids else -1


def compare_token_sequences(
    seq_a: List[int],
    seq_b: List[int]
) -> Dict[str, Any]:
    """
    Compare two token sequences (e.g. baseline vs ablated output).
    Returns which tokens differ and where the first divergence is.

    Used by the ablation diff view.
    """
    min_len = min(len(seq_a), len(seq_b))
    differences = []
    first_diff = None

    for i in range(min_len):
        if seq_a[i] != seq_b[i]:
            if first_diff is None:
                first_diff = i
            differences.append({
                "position": i,
                "token_a_id": seq_a[i],
                "token_b_id": seq_b[i],
                "token_a": decode_token_ids([seq_a[i]]),
                "token_b": decode_token_ids([seq_b[i]]),
            })

    return {
        "identical": len(differences) == 0,
        "num_differences": len(differences),
        "first_divergence_position": first_diff,
        "differences": differences,
        "len_a": len(seq_a),
        "len_b": len(seq_b),
        "length_differs": len(seq_a) != len(seq_b),
    }


def get_vocabulary_info() -> Dict[str, Any]:
    """
    Return info about GPT-2's vocabulary.
    Useful for the frontend's token explorer panel.
    """
    tokenizer = model_loader.get_tokenizer()
    vocab = tokenizer.get_vocab()

    # Sample some interesting tokens for display
    sample_tokens = {
        "common_words": ["the", "and", "is", "in", "of"],
        "punctuation": [".", ",", "!", "?", ":"],
        "numbers": ["0", "1", "2", "10", "100"],
    }

    sample_ids = {}
    for category, words in sample_tokens.items():
        sample_ids[category] = [
            {"word": w, "id": tokenizer.encode(w)[0]}
            for w in words
        ]

    return {
        "vocab_size": len(vocab),
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "encoding_type": "Byte Pair Encoding (BPE)",
        "sample_tokens": sample_ids,
    }