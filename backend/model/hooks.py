"""
hooks.py — PyTorch forward hooks to capture internal activations.

HOW HOOKS WORK:
  A "hook" is a function that gets called automatically during the
  model's forward pass. It's like tapping a wire — you don't change
  the signal, you just read it.

  model.register_forward_hook(fn)  →  fn runs after every layer
  model.register_forward_pre_hook(fn)  →  fn runs before every layer

This is how Anthropic's interpretability team captures what's
happening inside the model during inference.
"""

import torch
from typing import Dict, List, Tuple, Any


class ActivationCapture:
    """
    Captures activations from every attention layer during a forward pass.

    Usage:
        capture = ActivationCapture(model)
        with capture:
            output = model(input_ids)
        activations = capture.get_activations()
    """

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.activations: Dict[str, Any] = {}

    def _make_hook(self, layer_name: str):
        """Creates a hook function for a specific layer."""
        def hook_fn(module, input, output):
            # output from attention layer is a tuple:
            # (hidden_states, attention_weights, ...)
            if isinstance(output, tuple):
                hidden = output[0]          # shape: (batch, seq_len, hidden_dim)
                attn = output[1] if len(output) > 1 else None  # shape: (batch, heads, seq, seq)
            else:
                hidden = output
                attn = None

            self.activations[layer_name] = {
                "hidden_states": hidden.detach().cpu(),
                "attention_weights": attn.detach().cpu() if attn is not None else None,
            }
        return hook_fn

    def register(self):
        """Attach hooks to every transformer block's attention layer."""
        for i, block in enumerate(self.model.transformer.h):
            hook = block.attn.register_forward_hook(
                self._make_hook(f"layer_{i}")
            )
            self.hooks.append(hook)

    def remove(self):
        """Remove all hooks to clean up memory."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_activations(self) -> Dict[str, Any]:
        return self.activations

    def clear(self):
        self.activations.clear()

    # Context manager support: `with ActivationCapture(model) as cap:`
    def __enter__(self):
        self.clear()
        self.register()
        return self

    def __exit__(self, *args):
        self.remove()