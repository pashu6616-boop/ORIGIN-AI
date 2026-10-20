# ORIGIN AI v0.1

ORIGIN is a local research implementation for training a small decoder-only language model from random initialization. It includes a configurable Transformer, SentencePiece BPE tokenizer, local corpus preparation, checkpoint/resume support, short text generation, and a Streamlit interface. It uses no pretrained model weights and makes no paid API calls.

This is an experimental training system, not a production assistant or a GPT-level model. The default model has 2 Transformer layers, 96 hidden dimensions, 4 attention heads, a 128-token context window, and about 433,000 parameters. A short run on WikiText-2 demonstrates that the pipeline learns and resumes; it does not establish useful general-purpose capabilities.

## Project layout

- `origin/` — model, tokenizer, corpus preparation, training, checkpoints, evaluation helpers, local retrieval, and modality status contracts.
- `scripts/train.py` — train from scratch or resume a checkpoint on CPU or CUDA.
- `scripts/evaluate.py` — generate a short continuation from a checkpoint.
- `scripts/download_wikitext2.py` — download and verify the three WikiText-2 splits.
- `configs/` — model dimensions and training parameters.
- `app/main.py` — local Streamlit interface.
- `notebooks/ORIGIN_AI_Colab.ipynb` — Colab workflow for a GPU runtime, Drive persistence, smoke training, full training, resume, evaluation, and generation.
- `tests/` — unit and integration checks for model, data, tokenizer, retrieval, checkpoint, and app paths.

The existing `backend/` and `frontend/` directories and the historical `AI_USAGE_LOG.md` and `PROMPTS.md` belong to the separate Aether application. They are retained in the repository and are not used by ORIGIN.

## Local setup (Windows)

Use Python 3.11 or 3.12 and run these commands from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run `\.venv\Scripts\python.exe -m pip install -r requirements.txt` and use that interpreter for later commands.

## Run checks and a pipeline smoke test

```powershell
python -m pytest tests -q
python scripts/train.py --data demo_data/toy.txt --smoke-test --device cpu --checkpoint checkpoints\smoke.pt
python scripts/evaluate.py "ORIGIN" --checkpoint checkpoints\smoke.pt --device cpu --max-new-tokens 40
```

The toy file is only for verifying that training, checkpoint saving, loading, and generation execute. It is not a useful language-model training corpus.

## Train on WikiText-2

Download the corpus and preserve the official train, validation, and test splits:

```powershell
python scripts/download_wikitext2.py
```

The downloader verifies pinned SHA-256 digests, writes text files under `data/wikitext-2/`, and records the source and license details in `SOURCE.txt`. Review those terms before redistributing the dataset. The test split is not used by the training or validation commands.

Train on CPU:

```powershell
python scripts/train.py `
  --data data\wikitext-2\wiki.train.tokens `
  --validation-data data\wikitext-2\wiki.valid.tokens `
  --device cpu `
  --steps 1000 `
  --checkpoint checkpoints\wikitext2.pt
```

To continue that run, use the same corpus and checkpoint with `--resume`; the requested number of steps is added to the saved step:

```powershell
python scripts/train.py `
  --data data\wikitext-2\wiki.train.tokens `
  --validation-data data\wikitext-2\wiki.valid.tokens `
  --device cpu `
  --resume `
  --steps 500 `
  --checkpoint checkpoints\wikitext2.pt
```

Use `--device cuda` to require an available CUDA GPU, or `--device auto` to select CUDA when available and CPU otherwise. Checkpoint metadata includes model/training configuration, metrics, and step; the tokenizer is stored alongside the checkpoint. Resume rejects a changed corpus fingerprint or incompatible model configuration.

Generate a sample:

```powershell
python scripts/evaluate.py "The meaning of language" --checkpoint checkpoints\wikitext2.pt --device auto --max-new-tokens 60
```

## Google Colab

Open `notebooks/ORIGIN_AI_Colab.ipynb`, select a Tesla T4 GPU runtime, and run the cells in order. The notebook verifies CUDA, mounts Google Drive, installs the additional corpus/tokenizer dependencies without replacing Colab's CUDA-enabled PyTorch, and asks you to upload the project ZIP if the source tree is not already in `/content/ORIGIN-AI`. Training checkpoints, corpus files, caches, and logs are saved under `/content/drive/MyDrive/ORIGIN-AI/`. It runs the toy smoke test before WikiText-2 training, resumes only when the saved checkpoint and tokenizer are valid, and evaluates on the held-out validation split.

## Data, checkpoints, and secrets

Local datasets and token caches under `data/`, all `checkpoints/`, model files, environment files, credential files, and private-key files are excluded by `.gitignore`. Keep API keys and other secrets in local environment files; never commit them. Example environment files remain tracked.

## Scope and limitations

- Model weights start randomly; no pretrained weights, hosted model calls, or paid APIs are used.
- The default architecture and WikiText-2 corpus are intentionally small. Training step counts are experiments, not evidence of assistant-level competence.
- The document retriever returns matching local TXT/Markdown excerpts separately; the language model does not synthesize grounded answers from retrieved documents.
- Image input, audio input, and text-to-speech are explicitly unsupported in v0.1.
- The app is a local experimental interface. It does not provide a production deployment, safety evaluation, or general-purpose assistant guarantee.