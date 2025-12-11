# MCP-Chatbot

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)]()
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)]()
[![Build Status](https://img.shields.io/badge/build-unknown-lightgrey.svg)]()
[![Model Card](https://img.shields.io/badge/model-card-available-green.svg)]()

State-of-the-art, modular, reproducible chatbot platform for research and production. MCP-Chatbot provides tools to train, evaluate, and deploy large conversational models, with an emphasis on clear experiment tracking, safe defaults, and easy integration with downstream apps.

Table of contents
- About
- Highlights
- Architecture & components
- Quickstart
- Installation (dev & prod)
- Running (local & Docker)
- Example usage (Python & API)
- Training & evaluation
- Reproducing SOTA results
- Datasets & preprocessing
- Benchmarks & metrics (template)
- Safety, privacy & limitations
- Contributing
- Citation & license
- Contact

---

About
-----
MCP-Chatbot is a modular framework for building modern conversational AI. It bundles:
- A training/evaluation pipeline compatible with standard transformer backbones.
- Pre-configured model and tokenizer handling.
- An inference server (FastAPI) with production-ready configuration.
- Utilities for dataset ingestion, preprocessing, and experiment tracking.
- Safety, rate-limiting, and plugin hooks to extend behavior.

Built for researchers and engineers who want reproducible experiments and deployable results.

Highlights
----------
- Modular: swap tokenizer/model/training loop with minimal code changes.
- Reproducible: config-driven experiments (YAML/JSON) + logging + checkpoints.
- Extensible: plugin hooks for moderation, memory, and retrieval-augmentation.
- Production-ready API: containerized FastAPI server, health checks, Prometheus metrics.
- Focus on safety: moderation hooks, logging, and opt-in telemetry.

Architecture & components
-------------------------
- data/ — dataset ingestion & preprocessing pipelines (parsers, dedupers, tokenizers).
- models/ — model wrapper classes, adapter layers, and export utilities.
- training/ — training loop, scheduler, checkpointing, mixed-precision support.
- inference/ — FastAPI app, request handlers, rate limiting, and batching.
- eval/ — evaluation scripts, metrics, and human-eval harnesses.
- tools/ — experiment management, scripts, and helpers.
- docs/ — design docs and model cards.

A typical flow:
1. Prepare dataset using data/ preprocessors.
2. Configure experiment in experiments/*.yaml.
3. Train with training/ script, log to experiment tracking (MLflow/W&B).
4. Evaluate with eval/ and generate reports.
5. Deploy using inference/ with a Docker image.

Quickstart
----------
Clone, create a virtual environment, install, and run the dev server:

git clone https://github.com/DawoodTahir/MCP-Chatbot.git
cd MCP-Chatbot
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# start local inference server (development)
python -m inference.app --config configs/dev_inference.yaml

Open http://localhost:8000/docs for interactive API docs.

Installation
------------
Recommended: Python 3.8+

Development
pip install -e ".[dev]"          # installs test/dev dependencies (black, pytest, flake8)

Production / minimal
pip install -e .

Optional accelerated dependencies
- For GPU training with PyTorch: pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
- For FP16 training: apex or torch.cuda.amp (native amp recommended)

Docker
Build:
docker build -t mcp-chatbot:latest .

Run:
docker run --rm -p 8000:8000 -e MCP_CONFIG=/app/configs/prod_inference.yaml mcp-chatbot:latest

Configuration
-------------
All experiments and runtime behavior are configuration-driven (YAML). Example fields:
- model: type, pretrained_checkpoint, tokenizer
- training: batch_size, lr, epochs, mixed_precision
- data: dataset_path, max_seq_len, cache
- inference: max_tokens, top_k, top_p, temperature
- logging: wandb/project, mlflow/uri

Example config: configs/example_train.yaml

Running locally
---------------
Start an interactive Python session to run inference:

from models import ChatModel
model = ChatModel.from_pretrained("checkpoints/last")
print(model.chat("Hello, how are you?"))

API example (curl)
------------------
POST /v1/generate
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain the Pythagorean theorem in simple terms.","max_tokens":128}'

Python client example
---------------------
from inference.client import ChatClient
client = ChatClient("http://localhost:8000")
resp = client.generate("Summarize the causes of WWI", max_tokens=150)
print(resp.text)

Training & evaluation
---------------------
Train a model:

python -m training.run \
  --config configs/example_train.yaml \
  --output_dir outputs/exp01 \
  --seed 42

Recommended practices:
- Use mixed precision on GPU for speed/memory.
- Use gradient accumulation to simulate larger batches.
- Log checkpoints and metrics to an experiment tracker (W&B/MLflow).
- Keep a deterministic config and record the random seed.

Evaluation
python -m eval.evaluate --config configs/eval.yaml --ckpt outputs/exp01/checkpoint.pt

Supported metrics (configurable):
- Perplexity (PPL)
- BLEU / ROUGE (if applicable)
- Exact Match / F1 (task-specific)
- Human evaluation harness (pairwise comparisons, Likert scales)

Reproducing SOTA results
-----------------------
This repository provides a reproducible experiment template. To reproduce results:
1. Pin dependencies (pip freeze > requirements.txt).
2. Use the same config in experiments/ and the same checkpoint.
3. Note hardware differences (GPU model, mixed precision).
4. Share seeds and random state.

If you publish a paper claiming SOTA, include:
- Full config YAML
- Training logs (W&B/MLflow links)
- Checkpoint or model card with evaluation artifacts
- Exact dataset splits and preprocessing scripts

Datasets & preprocessing
-----------------------
We don't redistribute proprietary datasets. Example public datasets used in experiments:
- OpenAssistant Conversations
- ParlAI datasets
- MultiWOZ (task-oriented)
- HumanEval (code models)

Preprocessing steps:
1. normalize_unicode
2. canonicalize punctuation
3. deduplicate
4. segment dialogues into context/response pairs
5. tokenize and create model-specific features

Example:
python -m data.prepare --dataset path/to/raw --out data/processed --config configs/data_preproc.yaml

Benchmarks & metrics (template)
------------------------------
Replace the placeholders with real, reproducible numbers from your runs.

Model | Dataset | Tokens | Params | Hardware | PPL | BLEU | Latency (ms)
----- | ------- | ------ | ------ | -------- | --- | ---- | -------------
mcp-small (ours) | OA-conv-test | 10M | 125M | 1xA100 | 12.5 | 18.2 | 45
mcp-base (ours)  | OA-conv-test | 50M | 350M | 1xA100 | 8.9  | 24.7 | 120

Notes:
- Report mean and standard deviation over at least 3 seeds when possible.
- Share raw evaluation logs and code to compute metrics.

Safety, privacy & limitations
----------------------------
- This repository includes moderation utilities (rules-based + classifier hooks). They are helpers — not a full safety solution.
- Models may produce harmful or biased outputs. Do not deploy without a safety review.
- Logging can be disabled; ensure PII is not stored in logs when processing user data.
- If you add a retrieval-augmented component, ensure sources are verified and caching is privacy-aware.

Contributing
------------
Contributions are welcome. Suggested flow:
1. Fork and create a feature branch.
2. Run tests and linters: pytest && flake8
3. Open a PR with a clear description and tests where appropriate.
4. Add to CHANGELOG.md and update docs.

Please read CONTRIBUTING.md and CODE_OF_CONDUCT.md (add these files if missing).

Model Card & Responsible Use
----------------------------
We include a model card (docs/model_card.md) describing:
- Intended use
- Limitations and biases
- Training data provenance
- Evaluation and metrics
- Known failure modes
- Recommended mitigations

License
-------
This project is distributed under the Apache-2.0 License. See LICENSE for details.

Acknowledgements & citations
---------------------------
If you use MCP-Chatbot in published work, please cite:

- The MCP-Chatbot repo: DOI/URL (add when available)
- Transformer models:
  - Vaswani et al., "Attention is All You Need", 2017.
- Any pretrained model checkpoints used (HuggingFace model names / papers).

Security & Responsible Disclosure
---------------------------------
If you discover a security vulnerability, please report it privately to the maintainers at <your-email@example.com> and do not create a public issue. Include steps to reproduce, affected versions, and potential impact.

Contact
-------
Maintainer: DawoodTahir
GitHub: https://github.com/DawoodTahir/MCP-Chatbot

What's next
-----------
- Add full experiment artifacts (configs, checkpoints, logs) to the experiments/ folder.
- Populate model benchmarks with real numbers and hardware details.
- Add automated CI for tests and validation of configs.

Thank you for using MCP-Chatbot — contributions and feedback are highly appreciated!
