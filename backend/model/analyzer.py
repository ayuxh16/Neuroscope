"""
analyzer.py — Core MI analysis engine.

Given a text prompt, this module:
1. Tokenizes it
2. Runs GPT-2 with activation capture hooks
3. Extracts attention weights per layer per head
4. Returns structured data the frontend can visualize

KEY CONCEPTS IMPLEMENTED HERE:
  - Attention weights: For each token, how much does it "attend to"
    every other token? High weight = strong connection.
  - Per-head analysis: GPT-2 has 12 heads × 12 layers = 144 attention
    patterns. Each head learns a different "role".
  - Top tokens: Which input tokens most influenced the output?
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional

from model.loader import model_loader
from model.hooks import ActivationCapture


class AnalysisResult:
    """Structured result from analyzing a prompt."""

    def __init__(
        self,
        tokens: List[str],
        token_ids: List[int],
        attention_by_layer: List[Dict],   # [{layer, head, matrix}]
        top_output_tokens: List[Dict],    # [{token, prob}]
        generated_text: str,
    ):
        self.tokens = tokens
        self.token_ids = token_ids
        self.attention_by_layer = attention_by_layer
        self.top_output_tokens = top_output_tokens
        self.generated_text = generated_text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tokens": self.tokens,
            "token_ids": self.token_ids,
            "num_tokens": len(self.tokens),
            "num_layers": 12,
            "num_heads": 12,
            "attention_by_layer": self.attention_by_layer,
            "top_output_tokens": self.top_output_tokens,
            "generated_text": self.generated_text,
        }


def analyze_prompt(
    prompt: str,
    max_new_tokens: int = 10,
    ablated_heads: Optional[List[Dict]] = None,  # [{"layer": 0, "head": 3}]
) -> AnalysisResult:
    """
    Main analysis function.

    Args:
        prompt: The input text to analyze
        max_new_tokens: How many tokens to generate after the prompt
        ablated_heads: List of {layer, head} dicts to disable before running

    Returns:
        AnalysisResult with attention maps and generated text
    """

    model = model_loader.get_model()
    tokenizer = model_loader.get_tokenizer()
    device = model_loader.get_device()

    # ── STEP 1: TOKENIZE ──────────────────────────────────────────────
    encoding = tokenizer(prompt, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)

    # Decode each token individually to get human-readable labels
    tokens = [
        tokenizer.decode([tid]).strip()
        for tid in input_ids[0].tolist()
    ]

    # ── STEP 2: ABLATION (if requested) ───────────────────────────────
    # Ablation = zeroing out an attention head's output
    # This is how we test "what does this head do?"
    ablation_hooks = []
    if ablated_heads:
        for entry in ablated_heads:
            layer_idx = entry["layer"]
            head_idx = entry["head"]
            hook = _register_ablation_hook(model, layer_idx, head_idx)
            ablation_hooks.append(hook)

    # ── STEP 3: FORWARD PASS WITH HOOKS ───────────────────────────────
    try:
        with ActivationCapture(model) as capture:
            with torch.no_grad():
                output = model(
                    input_ids,
                    output_attentions=True,
                    output_hidden_states=True,
                )
        activations = capture.get_activations()
    finally:
        # Always remove ablation hooks
        for hook in ablation_hooks:
            hook.remove()

    # ── STEP 4: EXTRACT ATTENTION WEIGHTS ────────────────────────────
    # output.attentions = tuple of 12 tensors
    # Each tensor shape: (batch=1, heads=12, seq_len, seq_len)
    attention_by_layer = []

    for layer_idx, attn_tensor in enumerate(output.attentions):
        # Remove batch dim → (12, seq_len, seq_len)
        attn = attn_tensor[0].cpu().numpy()

        for head_idx in range(attn.shape[0]):
            head_matrix = attn[head_idx]  # (seq_len, seq_len)

            # Find the most "attended-to" token pair for this head
            flat_idx = np.argmax(head_matrix)
            from_tok = int(flat_idx // head_matrix.shape[1])
            to_tok = int(flat_idx % head_matrix.shape[1])

            attention_by_layer.append({
                "layer": layer_idx,
                "head": head_idx,
                # Flatten matrix for JSON (frontend rebuilds it)
                "matrix": head_matrix.tolist(),
                "max_attention_from": from_tok,
                "max_attention_to": to_tok,
                "max_attention_value": float(head_matrix.max()),
            })

    # ── STEP 5: GENERATE NEXT TOKENS ─────────────────────────────────
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # Greedy = deterministic
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(
        gen_ids[0][input_ids.shape[1]:],  # Only the NEW tokens
        skip_special_tokens=True
    )

    # ── STEP 6: TOP NEXT-TOKEN PROBABILITIES ─────────────────────────
    # What token did the model think was most likely next?
    logits = output.logits[0, -1, :]          # Last position logits
    probs = torch.softmax(logits, dim=-1)
    top_k = torch.topk(probs, k=10)

    top_output_tokens = [
        {
            "token": tokenizer.decode([tid.item()]).strip(),
            "token_id": tid.item(),
            "probability": round(prob.item(), 4),
        }
        for tid, prob in zip(top_k.indices, top_k.values)
    ]

    return AnalysisResult(
        tokens=tokens,
        token_ids=input_ids[0].tolist(),
        attention_by_layer=attention_by_layer,
        top_output_tokens=top_output_tokens,
        generated_text=generated_text,
    )


def _register_ablation_hook(model, layer_idx: int, head_idx: int):
    """
    Zeros out a specific attention head's output.

    This is activation patching / ablation — the core MI experiment.
    When you ablate head (layer=3, head=7) and the model stops doing
    indirect object identification, you've found the responsible circuit.
    """

    def ablation_hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            # hidden shape: (batch, seq_len, hidden_dim)
            # Each head occupies hidden_dim/num_heads = 64 dims
            head_dim = 64
            start = head_idx * head_dim
            end = start + head_dim
            # Zero out this head's contribution
            hidden[:, :, start:end] = 0.0
            return (hidden,) + output[1:]
        return output

    block = model.transformer.h[layer_idx]
    return block.attn.register_forward_hook(ablation_hook)