"""
main.py - NeuroScope FastAPI backend

Endpoints:
  GET  /health            -> Check if model is loaded
  GET  /model/info        -> Model architecture info
  GET  /vocabulary        -> GPT-2 vocabulary metadata
  POST /tokenize          -> Token metadata for a prompt
  POST /analyze           -> Attention maps + predictions + generation
  POST /ablate            -> Before/after head-ablation experiment

Run with:
  uvicorn main:app --reload --port 8000
"""

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model.loader import model_loader
from model.analyzer import analyze_prompt
from model.ablation import run_ablation_experiment
from utils.tokenizer import tokenize, get_vocabulary_info
from utils.formatting import format_analyze_response, format_probability_diff

# -- LOGGING ---------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ablation works by attaching hooks to the shared, singleton model.
# Without a lock, a concurrent request could run while another request's
# hooks are attached and silently get ablated results.
model_lock = threading.Lock()

_ORIGINS = [
    "http://localhost:3000", "http://127.0.0.1:3000",
    "http://localhost:3001", "http://127.0.0.1:3001",
] + [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]


# -- LIFESPAN (replaces deprecated on_event("startup")) ---------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    logger.info("NeuroScope API starting up...")
    try:
        with model_lock:
            analyze_prompt("Hello world", max_new_tokens=1)
        app.state.ready = True
        logger.info("Model warm and ready")
    except Exception:
        logger.exception("Warm-up failed")
    yield


# -- APP -------------------------------------------------------------
app = FastAPI(
    title="NeuroScope API",
    description="Mechanistic interpretability tool for GPT-2 visualization",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- REQUEST MODELS --------------------------------------------------
class HeadRef(BaseModel):
    layer: int = Field(ge=0, le=11)
    head: int = Field(ge=0, le=11)


class TokenizeRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)


class AnalyzeRequest(BaseModel):
    prompt: str = Field(
        min_length=1,
        max_length=500,
        json_schema_extra={"examples": ["The Eiffel Tower is located in"]},
    )
    max_new_tokens: int = Field(default=10, ge=1, le=50)
    ablated_heads: Optional[List[HeadRef]] = Field(
        default=None,
        json_schema_extra={"examples": [[{"layer": 3, "head": 7}]]},
    )


class AblationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    heads_to_ablate: List[HeadRef] = Field(
        min_length=1,
        json_schema_extra={"examples": [[{"layer": 3, "head": 7}, {"layer": 8, "head": 11}]]},
    )


def _heads_as_dicts(heads: Optional[List[HeadRef]]):
    """Existing model code expects plain dicts like {"layer": 3, "head": 7}."""
    return [h.model_dump() for h in heads] if heads else None


# -- ENDPOINTS -------------------------------------------------------
@app.get("/health")
def health_check():
    ready = bool(getattr(app.state, "ready", False))
    return {
        "status": "ok" if ready else "starting",
        "model": model_loader.model_name,
        "device": str(model_loader.get_device()),
        "ready": ready,
    }


@app.get("/model/info")
def model_info():
    """Architecture details so the frontend knows how many layers/heads to render."""
    return {
        "model_name": "GPT-2 Small",
        "parameters": "117M",
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "head_dim": 64,  # hidden_dim / num_heads
        "vocab_size": 50257,
        "max_context_length": 1024,
        "architecture": "Decoder-only Transformer",
    }


@app.get("/vocabulary")
def vocabulary_info():
    """Return GPT-2 vocabulary metadata and sample tokens."""
    return get_vocabulary_info()


@app.post("/tokenize")
def tokenize_text(req: TokenizeRequest):
    """Tokenize a prompt and return rich token metadata for the token visualizer."""
    try:
        return tokenize(req.prompt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Tokenization failed")
        raise HTTPException(status_code=500, detail="Internal error during tokenization")


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """
    Run GPT-2 on a prompt and return the tokenized input, attention weights
    for all 144 heads (12 layers x 12 heads), top next-token predictions and
    the generated continuation. Optionally ablate specific heads.

    Note: output depends on the model; run the prompt to see actual results.
    """
    logger.info("Analyzing prompt (%d chars)", len(req.prompt))
    start = time.time()

    try:
        with model_lock:
            result = analyze_prompt(
                prompt=req.prompt,
                max_new_tokens=req.max_new_tokens,
                ablated_heads=_heads_as_dicts(req.ablated_heads),
            )
        elapsed = round(time.time() - start, 2)
        logger.info("Analysis complete in %ss", elapsed)

        token_info = tokenize(req.prompt)

        raw = result.to_dict()
        raw["elapsed_seconds"] = elapsed
        formatted = format_analyze_response(raw, token_info["clean_tokens"])
        formatted["token_metadata"] = token_info["token_metadata"]
        return formatted

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail="Internal error during analysis")


@app.post("/ablate")
def ablate(req: AblationRequest):
    """
    Run a before/after ablation experiment: the model runs once as a baseline
    and once with the chosen heads zeroed out, and the response contains both
    outputs plus a probability diff. Ablation is a standard technique in
    mechanistic interpretability for testing what a component contributes.
    """
    logger.info("Running ablation: %d heads", len(req.heads_to_ablate))
    start = time.time()

    try:
        with model_lock:
            result = run_ablation_experiment(
                prompt=req.prompt,
                heads_to_ablate=_heads_as_dicts(req.heads_to_ablate),
            )
        elapsed = round(time.time() - start, 2)

        result["probability_diff"] = format_probability_diff(
            result.get("probability_diff", [])
        )
        result["elapsed_seconds"] = elapsed
        logger.info(
            "Ablation complete in %ss. Output changed: %s",
            elapsed,
            result.get("output_changed"),
        )
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Ablation failed")
        raise HTTPException(status_code=500, detail="Internal error during ablation")