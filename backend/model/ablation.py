"""
ablation.py — Run before/after ablation experiments.

Ablation experiment workflow:
  1. Run model normally → record output A
  2. Zero out head (layer=X, head=Y)
  3. Run model again → record output B
  4. Compare A and B → the difference reveals what that head does

This is the gold standard MI experiment technique.
"""

from typing import List, Dict, Any
from model.analyzer import analyze_prompt


def run_ablation_experiment(
    prompt: str,
    heads_to_ablate: List[Dict],  # [{"layer": 3, "head": 7}]
) -> Dict[str, Any]:
    """
    Run baseline vs ablated comparison.

    Returns both results side-by-side so the frontend
    can show a diff of what changed.
    """

    # Baseline run — no ablation
    baseline = analyze_prompt(prompt, max_new_tokens=15, ablated_heads=None)

    # Ablated run — zero out specified heads
    ablated = analyze_prompt(prompt, max_new_tokens=15, ablated_heads=heads_to_ablate)

    # Calculate how much the output probability distribution changed
    baseline_top = {t["token"]: t["probability"] for t in baseline.top_output_tokens}
    ablated_top = {t["token"]: t["probability"] for t in ablated.top_output_tokens}

    all_tokens = set(baseline_top) | set(ablated_top)
    prob_diff = [
        {
            "token": tok,
            "baseline_prob": baseline_top.get(tok, 0.0),
            "ablated_prob": ablated_top.get(tok, 0.0),
            "delta": round(ablated_top.get(tok, 0.0) - baseline_top.get(tok, 0.0), 4),
        }
        for tok in all_tokens
    ]
    prob_diff.sort(key=lambda x: abs(x["delta"]), reverse=True)

    return {
        "prompt": prompt,
        "ablated_heads": heads_to_ablate,
        "baseline": {
            "generated_text": baseline.generated_text,
            "top_tokens": baseline.top_output_tokens,
        },
        "ablated": {
            "generated_text": ablated.generated_text,
            "top_tokens": ablated.top_output_tokens,
        },
        "probability_diff": prob_diff[:10],  # Top 10 most changed tokens
        "output_changed": baseline.generated_text != ablated.generated_text,
    }