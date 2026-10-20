# Local datasets

Put UTF-8 `.txt`, `.md`, `.tokens`, or `.jsonl` training data here. JSONL records must have a string `text` field. ORIGIN normalizes Unicode and whitespace, removes exact duplicate lines using a disk-backed index, makes a deterministic train/validation split, and stores tokenized data in a memory-mapped cache. The configurable input limit defaults to 5 GiB. Local corpora, caches, and checkpoints are ignored by Git.

For a reproducible real-text experiment, run `python scripts/download_wikitext2.py`. It retrieves the maintained Salesforce AI Research WikiText-2 files from Hugging Face, verifies pinned SHA-256 digests, converts Parquet to text, and preserves train/validation/test in `data/wikitext-2/`. Review the included `SOURCE.txt` attribution and license details. Never train or tune on the test split.

`demo_data/toy.txt` is only a pipeline smoke-test fixture, not a training recommendation. Use a corpus you are authorized to train on for actual experiments.
