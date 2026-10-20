import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from origin.config import ModelConfig
from origin.model import DecoderTransformer
from origin.tokenizer import load_tokenizer
from origin.training.checkpoint import load_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Generate a short continuation from a local ORIGIN checkpoint.")
    parser.add_argument("prompt", nargs="?", default="ORIGIN")
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "latest.pt"))
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Inference device; auto selects CUDA when available.")
    args = parser.parse_args()
    if not 0 <= args.max_new_tokens <= 200: parser.error("max-new-tokens must be between 0 and 200")
    if args.device == "cuda" and not torch.cuda.is_available(): parser.error("CUDA was requested but is not available in this PyTorch runtime.")
    device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    path = Path(args.checkpoint); tokenizer = load_tokenizer(path.with_suffix(".tokenizer.json"))
    payload = torch.load(path, map_location=device, weights_only=False)
    model = DecoderTransformer(ModelConfig(**payload["model_config"])).to(device); load_checkpoint(path, model, device=device)
    ids = torch.tensor([tokenizer.encode(args.prompt) or [tokenizer.pad_id]], dtype=torch.long, device=device)
    print(tokenizer.decode(model.generate(ids, max_new_tokens=args.max_new_tokens, temperature=0.9, top_k=20, eos_token_id=getattr(tokenizer, "eos_id", None))[0].tolist()))

if __name__ == "__main__": main()
