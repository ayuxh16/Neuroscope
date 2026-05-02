# 🔬 NeurоScope
### *See inside the mind of AI*

A Mechanistic Interpretability tool that lets you visualize what's happening inside GPT-2's attention heads — and surgically disable them to discover what each circuit does.

Built to demonstrate AI safety research techniques used by Anthropic's interpretability team.

---

## 🧠 What It Does

| Feature | Description |
|---|---|
| **Attention Heatmap** | See how every token attends to every other token, per head |
| **Circuit Visualization** | Node graph of active attention circuits |
| **Head Ablation** | Disable specific attention heads and watch the output change |
| **Probability Diff** | See exactly which next-token probabilities shift after ablation |
| **Layer Explorer** | Navigate all 144 heads (12 layers × 12 heads) |

---

## 🛠️ Tech Stack

**Backend:** Python · FastAPI · PyTorch · Hugging Face Transformers  
**Frontend:** Next.js 14 · TypeScript · Tailwind CSS · D3.js  
**Model:** GPT-2 Small (117M parameters, open source)

---

## 🚀 Quick Start

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check model status |
| GET | `/model/info` | Model architecture details |
| POST | `/analyze` | Analyze a prompt, get attention maps |
| POST | `/ablate` | Run before/after ablation experiment |

### Example: Analyze a prompt
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The Eiffel Tower is located in", "max_new_tokens": 5}'
```

### Example: Ablation experiment
```bash
curl -X POST http://localhost:8000/ablate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The Eiffel Tower is located in",
    "heads_to_ablate": [{"layer": 3, "head": 7}]
  }'
```

---

## 🔬 Key MI Concepts Implemented

**Attention Capture** — PyTorch forward hooks tap into every attention layer without modifying the model

**Head Ablation** — Zero out a head's output vector to test its function

**Activation Patching** — Coming in Week 3: replace activations from one run into another

**Feature Visualization** — Coming in Week 5: Sparse Autoencoder for monosemantic features

---

## 📁 Structure

```
neuroscope/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── model/
│   │   ├── loader.py        # GPT-2 loader (singleton)
│   │   ├── hooks.py         # Activation capture hooks
│   │   ├── analyzer.py      # Core analysis engine
│   │   └── ablation.py      # Ablation experiments
│   └── requirements.txt
├── frontend/                # Next.js app (Week 2)
├── notebooks/               # Jupyter research notebooks
└── README.md
```

---

## 🎯 Relevance to Anthropic

This project directly implements techniques from Anthropic's published research:
- [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
- [Interpretability in the Wild](https://arxiv.org/abs/2211.00593)

---

*Built by Ayush Singh 2026*
