# ORIGIN AI v0.1

The repository contains Aether. ORIGIN is added alongside it as a separate local research system. It trains a subword tokenizer from local text and a compact decoder-only Transformer from random weights. It has no pretrained model downloads or paid API calls. Image and audio features are not implemented, and model quality depends on the authorized training corpus, training time, and hardware.

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

Streamlit prints the local app address. Before training, it labels the model as randomly initialized. After training it loads the checkpoint. Choose “Search supplied documents” and upload TXT/Markdown files to display separate excerpts and filenames. PDF ingestion is not included.

## Train on a real corpus

For a reproducible real-text experiment, optionally download WikiText-2, a public Wikipedia-derived language-modeling dataset maintained in the Salesforce Hugging Face repository. The downloader validates the training file digest and never overwrites existing destination files. It needs `pyarrow` to convert the published Parquet splits to plain text. Review the included source/license note before use or redistribution.

```powershell
python scripts/download_wikitext2.py
python scripts/train.py --data .\data\wikitext-2\wiki.train.tokens --validation-data .\data\wikitext-2\wiki.valid.tokens --steps 1000
python scripts/train.py --data .\data\wikitext-2\wiki.train.tokens --validation-data .\data\wikitext-2\wiki.valid.tokens --resume --steps 500
```

The train and validation splits remain separate; the official test file is not used. You can also point `--data` to an authorized local `.txt`, `.md`, `.tokens`, or `.jsonl` file or directory. JSONL records must contain a string `text` field. ORIGIN performs streaming Unicode/whitespace cleanup and exact duplicate removal, trains BPE only on training data, and caches token IDs in memory-mapped files. The default corpus cap is 5 GiB. Checkpoints save every 100 steps and include model/optimizer state, tokenizer, training config, metrics, and step. Resume uses the same corpus fingerprint and adds the requested number of steps.

A verified 100-step CPU run in this development environment used 21,331 train lines and 2,318 validation lines, cached 3,431,590 train subword tokens, and reported train loss 6.55 / validation loss 6.35 at step 100. The checkpoint resumed to step 102. This proves the pipeline trains and resumes on the real split; it is far too little training to imply useful assistant behavior.

The default model is 2 layers, 96 hidden units, 4 attention heads, and 128 subword tokens of context. Adjust `configs/model.json` and `configs/training.json` for your CPU. The CLI reports parameter/optimizer memory estimate; activations add memory based on context and batch size. A dataset can be licensed for research without being licensed for redistribution, so review its terms. WikiText-2 training demonstrates learning from real text, not useful general-purpose assistant quality.

For pipeline verification only, use `python scripts/train.py --data demo_data/toy.txt --smoke-test`. This fixture is not the default and must not be described as useful training.

## Optional Google Colab Free

See [colab.md](colab.md). Colab may disconnect and may be CPU-only. Drive is never mounted automatically. Its instructions include a notebook-generation plan with an explicit, user-run mount step. No paid Colab tier or billing is required.

## Scope and limitations

`docs/architecture.md` describes the model and retrieval boundary. The keyword retriever displays matching excerpts and does not teach the language model. Generic multimodal message/context types are future interfaces only; image input, audio input, and text-to-speech return explicit unsupported status. The CPU smoke test is a pipeline check, not evidence of useful assistant quality.
