# Optional Google Colab Free workflow

Colab Free is optional, can disconnect, and may offer CPU only. It is not a required service. In a Python runtime, upload a reviewed copy of the project or clone the reviewed feature branch. Install from `requirements.txt` as needed. Check `torch.cuda.is_available()` and use CPU if it is false; the project CLI deliberately targets CPU for portability.

For a real text run, use the supplied WikiText-2 downloader or your own authorized corpus:

```bash
python scripts/download_wikitext2.py
python scripts/train.py --data data/wikitext-2/wiki.train.tokens --validation-data data/wikitext-2/wiki.valid.tokens --steps 1000 --checkpoint checkpoints/wikitext2.pt
python scripts/evaluate.py --checkpoint checkpoints/wikitext2.pt --prompt "The meaning of language"
```

The test split is preserved and is not used during training or validation. The tokenizer is learned from the training split only. `--smoke-test` with `demo_data/toy.txt` verifies code paths only; it does not produce a useful language model. Colab Free can disconnect and may offer CPU only. For durable storage, add and deliberately run a separate cell with `from google.colab import drive; drive.mount('/content/drive')`, review the mount prompt, then pass a folder you selected under `/content/drive/MyDrive/` to `--checkpoint`. Drive is never mounted automatically. After a disconnect, rerun with `--resume --checkpoint <same path>` and the same corpus paths.

## Notebook generation plan

1. Upload or clone the reviewed project.
2. Check Python, PyTorch, and accelerator availability, and display the selected device.
3. Upload or select an authorized UTF-8 corpus and validate its size.
4. Run the CPU smoke command and then a configurable small experiment.
5. Keep checkpoints ephemeral by default; provide a separate user-run Drive mount step.
6. Allow resuming from the selected checkpoint path after disconnection and display saved metrics.
