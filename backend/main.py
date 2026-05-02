"""
main.py — NeuроScope FastAPI backend

Endpoints:
  GET  /health            → Check if model is loaded
  POST /analyze           → Run full MI analysis on a prompt
  POST /ablate            → Run before/after ablation experiment
  GET  /model/info        → Model architecture info

Run with:
  uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import time

from model.loader import model_loader
from model.analyzer import analyze_prompt
from model.ablation import run_ablation_experiment

# ── LOGGING ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── APP ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="NeuроScope API",
    description="Mechanistic Interpretability tool for GPT-2 visualization",
    version="1.0.0",
)

# Allow frontend (Next.js on port 3000) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── REQUEST / RESPONSE MODELS ─────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=500,
        example="The Eiffel Tower is located in"
    )
    max_new_tokens: int = Field(default=10, ge=1, le=50)
    ablated_heads: Optional[List[Dict]] = Field(
        default=None,
        example=[{"layer": 3, "head": 7}]
    )

class AblationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500)
    heads_to_ablate: List[Dict] = Field(
        ...,
        example=[{"layer": 3, "head": 7}, {"layer": 8, "head": 11}]
    )


# ── ENDPOINTS ─────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Check if the API and model are ready."""
    return {
        "status": "ok",
        "model": model_loader.model_name,
        "device": str(model_loader.get_device()),
        "ready": True,
    }


@app.get("/model/info")
def model_info():
    """
    Return GPT-2 architecture details.
    Useful for the frontend to know how many layers/heads to render.
    """
    return {
        "model_name": "GPT-2 Small",
        "parameters": "117M",
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "head_dim": 64,          # hidden_dim / num_heads
        "vocab_size": 50257,
        "max_context_length": 1024,
        "architecture": "Decoder-only Transformer",
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """
    Core endpoint: Run GPT-2 on a prompt and return:
      - Tokenized input
      - Attention weights for all 144 heads (12 layers × 12 heads)
      - Top predicted next tokens with probabilities
      - Generated continuation text

    Optionally ablate specific heads to see how output changes.

    Example prompt: "The Eiffel Tower is located in"
    Expected output: " Paris"
    Try ablating head (3, 7) — known to be involved in factual recall!
    """
    logger.info(f"Analyzing prompt: '{req.prompt[:50]}...'")
    start = time.time()

    try:
        result = analyze_prompt(
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            ablated_heads=req.ablated_heads,
        )
        elapsed = round(time.time() - start, 2)
        logger.info(f"Analysis complete in {elapsed}s")

        data = result.to_dict()
        data["elapsed_seconds"] = elapsed
        return data

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ablate")
def ablate(req: AblationRequest):
    """
    Run a before/after ablation experiment.

    Runs the model TWICE:
      1. Baseline (no changes)
      2. With specified heads zeroed out

    Returns both outputs + a diff showing what changed.

    This is how Anthropic researchers identify which circuits
    are responsible for specific model behaviors.
    """
    logger.info(f"Running ablation: {len(req.heads_to_ablate)} heads on '{req.prompt[:40]}'")
    start = time.time()

    try:
        result = run_ablation_experiment(
            prompt=req.prompt,
            heads_to_ablate=req.heads_to_ablate,
        )
        elapsed = round(time.time() - start, 2)
        result["elapsed_seconds"] = elapsed
        logger.info(f"Ablation complete in {elapsed}s. Output changed: {result['output_changed']}")
        return result

    except Exception as e:
        logger.error(f"Ablation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── STARTUP ───────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """
    Pre-warm the model on startup.
    This ensures the first real request is fast.
    """
    logger.info("NeuроScope API starting up...")
    logger.info("Pre-warming model with dummy forward pass...")

    try:
        _ = analyze_prompt("Hello world", max_new_tokens=1)
        logger.info("✅ Model warm and ready!")
    except Exception as e:
        logger.error(f"❌ Warm-up failed: {e}")