import json
from pathlib import Path
import os
import tempfile
import torch

def save_checkpoint(path, model, optimizer, tokenizer, training_config: dict, metrics: dict, step: int) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_config": model.config.to_dict(), "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "training_config": training_config, "metrics": metrics, "step": step}
    with tempfile.NamedTemporaryFile(prefix=f"{target.stem}-", suffix=".tmp", dir=target.parent, delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)
    tokenizer.save(target.with_suffix(".tokenizer.json"))
    target.with_suffix(".metadata.json").write_text(json.dumps({"model_config": model.config.to_dict(), "training_config": training_config, "metrics": metrics, "step": step}, indent=2), encoding="utf-8")

def load_checkpoint(path, model, optimizer=None, device: str = "cpu") -> dict:
    source = Path(path)
    if not source.is_file(): raise FileNotFoundError(f"Checkpoint not found: {source}")
    payload = torch.load(source, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model_config" not in payload or "model_state" not in payload:
        raise ValueError("Invalid checkpoint structure.")
    if payload["model_config"] != model.config.to_dict(): raise ValueError("Checkpoint model configuration does not match the model.")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        if "optimizer_state" not in payload: raise ValueError("Checkpoint has no optimizer state to resume.")
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload
