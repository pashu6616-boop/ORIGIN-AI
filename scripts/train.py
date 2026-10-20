from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from origin.config import ModelConfig
from origin.model import DecoderTransformer
from origin.tokenizer import CharacterTokenizer
from origin.training.checkpoint import load_checkpoint, save_checkpoint
from origin.training.data import load_texts, split_texts
from origin.training.trainer import seed_everything, train_steps

def main():
    parser = argparse.ArgumentParser(description="Train ORIGIN's small Transformer from random initialization.")
    parser.add_argument("--data", default=str(ROOT / "demo_data" / "toy.txt"))
    parser.add_argument("--model-config", default=str(ROOT / "configs" / "model.json"))
    parser.add_argument("--training-config", default=str(ROOT / "configs" / "training.json"))
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "latest.pt"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    try:
        texts = load_texts(args.data)
        training = json.loads(Path(args.training_config).read_text(encoding="utf-8"))
        train_texts, val_texts = split_texts(texts, validation_fraction=0.1 if args.smoke_test else training["validation_fraction"], seed=training["seed"])
        tokenizer = CharacterTokenizer.from_texts(texts)
        if tokenizer.vocab_size > 4096:
            raise ValueError("Character vocabulary exceeds 4096 entries; use a smaller or more normalized text dataset.")
        train_ids = tokenizer.encode("\n".join(train_texts) + "\n")
        val_ids = tokenizer.encode("\n".join(val_texts) + "\n") if val_texts else []
        model_values = json.loads(Path(args.model_config).read_text(encoding="utf-8")); model_values["vocab_size"] = tokenizer.vocab_size
        config = ModelConfig(**model_values); config.validate()
        if args.smoke_test: training.update(steps=4, batch_size=2, eval_batches=1)
        validation_fallback = len(val_ids) <= config.block_size
        if validation_fallback:
            val_ids = []
            training["validation_fallback_to_training_data"] = True
        seed_everything(int(training["seed"]))
        model = DecoderTransformer(config); optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]))
        start_step = 0
        if args.resume:
            checkpoint = Path(args.checkpoint)
            prior = load_checkpoint(checkpoint, model, optimizer, device="cpu")
            tokenizer_path = checkpoint.with_suffix(".tokenizer.json")
            if not tokenizer_path.is_file(): raise ValueError("Checkpoint tokenizer metadata is missing.")
            if CharacterTokenizer.load(tokenizer_path).vocabulary != tokenizer.vocabulary: raise ValueError("Dataset tokenizer differs from checkpoint; use the original dataset or start fresh.")
            start_step = int(prior.get("step", 0))
        if min(len(train_ids), len(val_ids) or len(train_ids)) <= config.block_size: raise ValueError("Not enough tokens for block_size; add text or reduce configs/model.json block_size.")
        metrics = train_steps(model, optimizer, train_ids, val_ids, steps=int(training["steps"]), block_size=config.block_size, batch_size=int(training["batch_size"]), seed=int(training["seed"]), eval_batches=int(training["eval_batches"]))
        target = Path(args.checkpoint)
        save_checkpoint(target, model, optimizer, tokenizer, training, metrics, start_step + int(training["steps"]))
        print(json.dumps({"status":"ok","device":"cpu","unique_lines":len(texts),"vocab_size":tokenizer.vocab_size,"step":start_step+int(training["steps"]),"train_loss":metrics["train_loss"][-1],"validation_loss":metrics["validation_loss"][-1],"validation_fallback_to_training_data":validation_fallback,"checkpoint":str(target)}, indent=2))
    except (FileNotFoundError, ValueError, MemoryError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

if __name__ == "__main__": main()
