# ORIGIN AI v0.1

The existing repository contains Aether. ORIGIN is added alongside it as a separate research prototype. ORIGIN uses a tiny Transformer with random initialization and separate keyword search over local documents. It is not equivalent to ChatGPT, has no general knowledge before training, and has no working image or audio capabilities. It does not call a paid API or download pretrained weights.

## Windows 11 setup

Install Python 3.11 or 3.12 (3.13 is also supported by the current dependencies). From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If activation is blocked by PowerShell policy, run `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and prefix later Python commands with `.\.venv\Scripts\python.exe`.

## Run and verify

```powershell
python -m pytest tests -q
python scripts/train.py --data demo_data/toy.txt --smoke-test
python scripts/evaluate.py "ORIGIN"
streamlit run app/main.py
```

Streamlit prints the local app address. Before training, it labels the model as randomly initialized. After smoke training it loads the checkpoint. Choose “Search supplied documents” and upload TXT/Markdown files to display separate excerpts and filenames. PDF and recursive directory scanning are not included.

## Training experiment

Place authorized UTF-8 text in a local `.txt` file, one example/document per line. Then run:

```powershell
python scripts/train.py --data .\data\my_corpus.txt
python scripts/train.py --data .\data\my_corpus.txt --resume
```

Adjust `configs/model.json` and `configs/training.json` first. The shipped model is 1 layer, 32 hidden units, 2 attention heads, and 64 character context. Checkpoint weights, tokenizer, config, metrics, and optimizer state are saved under the ignored `checkpoints/` directory. Resume requires the same dataset vocabulary and model shape. The loader rejects input files over 25 MB, removes duplicate non-empty lines, and makes a seeded line-level split. Output quality will be very poor on the tiny demonstration data. Memory use changes with vocabulary, context, batch size, and hidden width; lower those values on constrained computers.

## Optional Google Colab Free

See [colab.md](colab.md). Colab may disconnect and may be CPU-only. Drive is never mounted automatically. Its instructions include a notebook-generation plan with an explicit, user-run mount step. No paid Colab tier or billing is required.

## Scope and limitations

`docs/architecture.md` describes the model and retrieval boundary. The keyword retriever displays matching excerpts and does not teach the language model. Generic multimodal message/context types are future interfaces only; image input, audio input, and text-to-speech return explicit unsupported status. The CPU smoke test is a pipeline check, not evidence of useful assistant quality.
