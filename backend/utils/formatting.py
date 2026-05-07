"""
formatting.py — Format raw model outputs into clean, frontend-ready JSON.

WHY THIS FILE EXISTS:
  The raw outputs from PyTorch are:
    - Tensors with shape (batch, heads, seq, seq)
    - Numpy arrays with 6+ decimal places
    - Nested tuples that JSON can't serialize

  The frontend needs:
    - Clean flat dicts
    - Rounded numbers (saves bandwidth)
    - Normalized values (0.0 to 1.0 range)
    - Human-readable labels

  This file is the "translator" between ML world and web world.
"""

import numpy as np
from typing import List, Dict, Any, Optional


# ── NUMBER FORMATTING ─────────────────────────────────────────────────

def round_matrix(matrix: List[List[float]], decimals: int = 4) -> List[List[float]]:
    """
    Round a 2D matrix to N decimal places.
    Reduces JSON payload size significantly.
    e.g. 0.023847291 → 0.0238
    """
    return [[round(val, decimals) for val in row] for row in matrix]


def normalize_matrix(matrix: List[List[float]]) -> List[List[float]]:
    """
    Normalize values to [0, 1] range using min-max normalization.
    Used for coloring the attention heatmap.

    Without this, some heads have very peaked distributions
    (one cell = 0.99) while others are flat (all cells = 0.08).
    Normalization makes all heatmaps visually comparable.
    """
    flat = [val for row in matrix for val in row]
    min_val = min(flat)
    max_val = max(flat)
    diff = max_val - min_val

    if diff == 0:
        # All values are equal — flat distribution
        return [[0.5 for _ in row] for row in matrix]

    return [
        [(val - min_val) / diff for val in row]
        for row in matrix
    ]


def entropy_of_distribution(matrix_row: List[float]) -> float:
    """
    Calculate Shannon entropy of an attention distribution.

    LOW entropy  → head is very focused (attending to 1-2 tokens)
    HIGH entropy → head is diffuse (attending equally to all tokens)

    Focused heads are more interpretable — they're doing something specific.
    """
    arr = np.array(matrix_row, dtype=np.float64)
    # Avoid log(0)
    arr = np.clip(arr, 1e-10, 1.0)
    return float(-np.sum(arr * np.log2(arr)))


# ── ATTENTION FORMATTING ──────────────────────────────────────────────

def format_attention_head(
    layer: int,
    head: int,
    matrix: List[List[float]],
    tokens: List[str],
) -> Dict[str, Any]:
    """
    Format a single attention head's data for the frontend.

    Input:  raw attention matrix (seq_len × seq_len)
    Output: clean dict with:
      - rounded matrix
      - normalized matrix (for coloring)
      - per-row entropy (how focused is each token's attention?)
      - top attended pairs (which token most strongly attends to which?)
      - human-readable label
    """
    seq_len = len(matrix)

    # Round for smaller payload
    rounded = round_matrix(matrix, decimals=4)

    # Normalize for heatmap colors
    normalized = normalize_matrix(rounded)

    # Entropy per row (one entropy value per token)
    entropies = [
        round(entropy_of_distribution(row), 3)
        for row in rounded
    ]

    # Find top 5 strongest attention pairs
    pairs = []
    for from_idx, row in enumerate(rounded):
        for to_idx, val in enumerate(row):
            pairs.append((val, from_idx, to_idx))
    pairs.sort(reverse=True)
    top_pairs = [
        {
            "from_token": tokens[f] if f < len(tokens) else f"[{f}]",
            "to_token": tokens[t] if t < len(tokens) else f"[{t}]",
            "from_idx": f,
            "to_idx": t,
            "weight": v,
        }
        for v, f, t in pairs[:5]
    ]

    # Mean attention entropy across all rows
    mean_entropy = round(float(np.mean(entropies)), 3)
    max_entropy = round(float(np.log2(seq_len)), 3) if seq_len > 0 else 0

    # Classify head focus style
    focus_ratio = mean_entropy / max_entropy if max_entropy > 0 else 0
    if focus_ratio < 0.3:
        focus_style = "very_focused"      # Likely doing specific task
    elif focus_ratio < 0.6:
        focus_style = "moderate"
    else:
        focus_style = "diffuse"           # Attending broadly

    return {
        "layer": layer,
        "head": head,
        "label": f"L{layer}H{head}",      # e.g. "L3H7"
        "matrix": rounded,
        "matrix_normalized": normalized,
        "seq_len": seq_len,
        "entropies": entropies,
        "mean_entropy": mean_entropy,
        "max_possible_entropy": max_entropy,
        "focus_style": focus_style,
        "top_attention_pairs": top_pairs,
    }


def format_all_attention_heads(
    attention_by_layer: List[Dict],
    tokens: List[str],
) -> Dict[str, Any]:
    """
    Format all 144 attention heads (12 layers × 12 heads) together.

    Also computes cross-head statistics:
      - Which layer is most active overall?
      - Which head has the highest peak attention?
      - Summary stats per layer
    """
    formatted_heads = []
    layer_summaries: Dict[int, Dict] = {}

    for entry in attention_by_layer:
        layer = entry["layer"]
        head = entry["head"]
        matrix = entry["matrix"]

        formatted = format_attention_head(layer, head, matrix, tokens)
        formatted_heads.append(formatted)

        # Aggregate per-layer stats
        if layer not in layer_summaries:
            layer_summaries[layer] = {
                "layer": layer,
                "heads": [],
                "mean_entropy_values": [],
                "focus_styles": [],
            }
        layer_summaries[layer]["heads"].append(f"L{layer}H{head}")
        layer_summaries[layer]["mean_entropy_values"].append(formatted["mean_entropy"])
        layer_summaries[layer]["focus_styles"].append(formatted["focus_style"])

    # Finalize layer summaries
    layer_summary_list = []
    for layer_idx in sorted(layer_summaries.keys()):
        s = layer_summaries[layer_idx]
        entropies = s["mean_entropy_values"]
        layer_summary_list.append({
            "layer": layer_idx,
            "avg_entropy": round(float(np.mean(entropies)), 3),
            "min_entropy": round(float(np.min(entropies)), 3),
            "max_entropy": round(float(np.max(entropies)), 3),
            "num_focused_heads": s["focus_styles"].count("very_focused"),
            "num_diffuse_heads": s["focus_styles"].count("diffuse"),
        })

    # Global most interesting head (lowest entropy = most focused = most interpretable)
    most_focused = min(formatted_heads, key=lambda h: h["mean_entropy"])
    most_diffuse = max(formatted_heads, key=lambda h: h["mean_entropy"])

    return {
        "heads": formatted_heads,
        "total_heads": len(formatted_heads),
        "layer_summaries": layer_summary_list,
        "most_focused_head": most_focused["label"],
        "most_diffuse_head": most_diffuse["label"],
    }


# ── TOKEN PROBABILITY FORMATTING ──────────────────────────────────────

def format_top_tokens(
    top_tokens: List[Dict],
    show_percentage: bool = True,
) -> List[Dict]:
    """
    Format next-token probabilities for display.

    Adds:
      - Percentage string (e.g. "34.2%")
      - Bar width for a visual probability bar (0–100)
      - Rank label
    """
    formatted = []
    max_prob = top_tokens[0]["probability"] if top_tokens else 1.0

    for rank, token_data in enumerate(top_tokens):
        prob = token_data["probability"]
        formatted.append({
            **token_data,
            "rank": rank + 1,
            "percentage": f"{prob * 100:.1f}%",
            "bar_width": round((prob / max_prob) * 100, 1),  # relative bar
            "is_top": rank == 0,
        })

    return formatted


# ── ABLATION DIFF FORMATTING ──────────────────────────────────────────

def format_probability_diff(prob_diff: List[Dict]) -> List[Dict]:
    """
    Format the before/after probability changes for the diff view.

    Adds color coding:
      "increased" → green  (ablating this head HELPED this token)
      "decreased" → red    (ablating this head HURT this token)
      "unchanged" → gray
    """
    formatted = []
    for entry in prob_diff:
        delta = entry["delta"]

        if delta > 0.005:
            direction = "increased"
            color = "green"
        elif delta < -0.005:
            direction = "decreased"
            color = "red"
        else:
            direction = "unchanged"
            color = "gray"

        formatted.append({
            **entry,
            "direction": direction,
            "color": color,
            "baseline_pct": f"{entry['baseline_prob'] * 100:.1f}%",
            "ablated_pct": f"{entry['ablated_prob'] * 100:.1f}%",
            "delta_pct": f"{delta * 100:+.1f}%",  # + sign for positive
            "abs_delta": abs(delta),
        })

    # Sort by absolute change size (most impacted first)
    formatted.sort(key=lambda x: x["abs_delta"], reverse=True)
    return formatted


# ── FULL RESPONSE FORMATTER ───────────────────────────────────────────

def format_analyze_response(
    raw_result: Dict[str, Any],
    tokens: List[str],
) -> Dict[str, Any]:
    """
    Master formatter — takes raw analyzer output and returns
    the complete, frontend-ready response.

    Called by main.py before returning from /analyze endpoint.
    """
    return {
        # Token info
        "tokens": tokens,
        "num_tokens": len(tokens),

        # Formatted attention (all 144 heads)
        "attention": format_all_attention_heads(
            raw_result.get("attention_by_layer", []),
            tokens,
        ),

        # Formatted next-token predictions
        "top_output_tokens": format_top_tokens(
            raw_result.get("top_output_tokens", [])
        ),

        # Generated continuation
        "generated_text": raw_result.get("generated_text", ""),

        # Metadata
        "model": "GPT-2 Small",
        "elapsed_seconds": raw_result.get("elapsed_seconds", 0),
    }