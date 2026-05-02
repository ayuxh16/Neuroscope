"""
loader.py — Loads GPT-2 and keeps it in memory for fast inference.
We use GPT-2 (small, 117M params) because:
  - It's open source and free
  - Small enough to run on CPU
  - Well-studied: lots of MI research done on it
  - Same transformer architecture as larger models
"""

import torch
from transformers import GPT2Model, GPT2Tokenizer, GPT2LMHeadModel
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Singleton class that holds the loaded model.
    We load once and reuse — loading takes ~5 seconds,
    but inference takes ~0.1 seconds.
    """

    _instance: Optional["ModelLoader"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        logger.info("Loading GPT-2 model and tokenizer...")

        # GPT-2 small = 12 layers, 12 attention heads, 768 hidden dim
        self.model_name = "gpt2"
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(
            self.model_name,
            output_attentions=True,       # <-- This gives us attention weights
            output_hidden_states=True,    # <-- This gives us layer activations
        )

        # Set pad token (GPT-2 doesn't have one by default)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Eval mode = no dropout, deterministic outputs
        self.model.eval()

        # Move to GPU if available, else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        logger.info(f"Model loaded on {self.device}")
        logger.info(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
        logger.info(f"Layers: 12, Attention heads: 12, Hidden dim: 768")

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_device(self):
        return self.device


# Global instance — imported everywhere
model_loader = ModelLoader()