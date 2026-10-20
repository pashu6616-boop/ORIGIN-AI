from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from origin.config import ModelConfig
from origin.model import DecoderTransformer
from origin.tokenizer import SubwordTokenizer, load_tokenizer
from origin.training.checkpoint import load_checkpoint, save_checkpoint
from origin.training.data import prepare_corpus, tokenize_to_cache
from origin.training.trainer import seed_everything, train_steps

def main():
    parser = argparse.ArgumentParser(description="Train ORIGIN from random initialization on a local text corpus.")
    parser.add_argument("--data", default=str(ROOT / "data"), help="A .txt/.md/.jsonl file or directory of authorized text files.")
    parser.add_argument("--validation-data", help="Optional separate validation file or directory; preserves a dataset's official split.")
    parser.add_argument("--model-config", default=str(ROOT / "configs" / "model.json"))
    parser.add_argument("--training-config", default=str(ROOT / "configs" / "training.json"))
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "latest.pt"))
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--steps", type=int, help="Training steps for this invocation (additional steps when resuming).")
    parser.add_argument("--vocab-size", type=int, help="Override the BPE tokenizer vocabulary size.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Use smaller settings; choose demo_data/toy.txt explicitly for a pipeline-only check.")
    args = parser.parse_args()
    try:
        cache_dir = Path(args.cache_dir)
        checkpoint = Path(args.checkpoint)
        training = json.loads(Path(args.training_config).read_text(encoding="utf-8"))
        model_values = json.loads(Path(args.model_config).read_text(encoding="utf-8"))
        if args.smoke_test:
            model_values.update(block_size=32, n_embd=32, n_head=2, n_layer=1, dropout=0.0)
            training.update(steps=4, batch_size=2, eval_batches=1, eval_interval=2, save_every_steps=2, vocab_size=512)
        if args.steps is not None:
            if args.steps < 1: raise ValueError("--steps must be positive.")
            training["steps"] = args.steps
        if args.vocab_size is not None: training["vocab_size"] = args.vocab_size
        if args.validation_data:
            train_manifest = prepare_corpus(args.data, cache_dir / "train", validation_fraction=0.0, seed=int(training["seed"]))
            validation_manifest = prepare_corpus(args.validation_data, cache_dir / "validation", validation_fraction=0.0, seed=int(training["seed"]))
            manifest = {**train_manifest, "validation_lines": validation_manifest["train_lines"], "duplicates": train_manifest["duplicates"] + validation_manifest["duplicates"], "validation_path": validation_manifest["train_path"], "fingerprint": hashlib.sha256((train_manifest["fingerprint"] + validation_manifest["fingerprint"]).encode()).hexdigest()}
        else:
            manifest = prepare_corpus(args.data, cache_dir / "corpus", validation_fraction=float(training["validation_fraction"]), seed=int(training["seed"]))
        tokenizer_path = checkpoint.with_suffix(".tokenizer.json")
        if args.resume:
            if not tokenizer_path.is_file(): raise FileNotFoundError("Checkpoint tokenizer metadata is missing; cannot resume safely.")
            tokenizer = load_tokenizer(tokenizer_path)
        else:
            tokenizer = SubwordTokenizer.train(manifest["train_path"], vocab_size=int(training["vocab_size"]), character_coverage=float(training["character_coverage"]))
        train_ids = tokenize_to_cache(tokenizer, manifest["train_path"], cache_dir / "train.bin", manifest["fingerprint"])
        val_ids = tokenize_to_cache(tokenizer, manifest["validation_path"], cache_dir / "validation.bin", manifest["fingerprint"]) if manifest["validation_path"] else []
        config = ModelConfig(**model_values, vocab_size=tokenizer.vocab_size); config.validate()
        validation_fallback = val_ids is None or len(val_ids) <= config.block_size
        if validation_fallback:
            val_ids = []
            training["validation_fallback_to_training_data"] = True
        training["corpus_fingerprint"] = manifest["fingerprint"]
        torch.set_num_threads(max(1, min(int(training.get("cpu_threads", 2)), os.cpu_count() or 1)))
        seed_everything(int(training["seed"]))
        model = DecoderTransformer(config)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]))
        start_step = 0
        if args.resume:
            prior = load_checkpoint(checkpoint, model, optimizer, device="cpu")
            if prior.get("training_config", {}).get("corpus_fingerprint") != manifest["fingerprint"]: raise ValueError("The current corpus differs from the checkpoint corpus; use the original corpus or start a new run.")
            start_step = int(prior.get("step", 0))
        if len(train_ids) <= config.block_size: raise ValueError(f"Training corpus has {len(train_ids)} tokens; block_size={config.block_size}. Add more text or reduce block_size.")
        if val_ids is not None and len(val_ids) <= config.block_size: validation_fallback = True; val_ids = []
        metrics = {"train_loss": [], "validation_loss": []}
        remaining = int(training["steps"])
        save_interval = max(1, int(training["save_every_steps"]))
        while remaining:
            segment = min(save_interval, remaining)
            partial = train_steps(model, optimizer, train_ids, val_ids, steps=segment, block_size=config.block_size, batch_size=int(training["batch_size"]), seed=int(training["seed"]) + start_step, eval_batches=int(training["eval_batches"]), eval_interval=int(training["eval_interval"]))
            metrics["train_loss"].extend(partial["train_loss"])
            metrics["validation_loss"].extend(partial["validation_loss"])
            start_step += segment; remaining -= segment
            save_checkpoint(checkpoint, model, optimizer, tokenizer, training, metrics, start_step)
            print(f"step {start_step}: train_loss={metrics['train_loss'][-1]:.4f}; validation_loss={metrics['validation_loss'][-1] if metrics['validation_loss'] else 'pending'}", flush=True)
        print(json.dumps({"status":"ok","device":"cpu","source_files":len(manifest["source_files"]),"train_lines":manifest["train_lines"],"validation_lines":manifest["validation_lines"],"duplicate_lines_removed":manifest["duplicates"],"invalid_json_lines":manifest["invalid_json_lines"],"train_tokens":len(train_ids),"vocab_size":tokenizer.vocab_size,"parameters":parameter_count,"estimated_optimizer_and_parameter_mib":round(parameter_count * 16 / (1024 ** 2), 2),"step":start_step,"train_loss":metrics["train_loss"][-1],"validation_loss":metrics["validation_loss"][-1] if metrics["validation_loss"] else None,"validation_fallback_to_training_data":validation_fallback,"checkpoint":str(checkpoint)}, indent=2))
    except (FileNotFoundError, ValueError, MemoryError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

if __name__ == "__main__": main()
