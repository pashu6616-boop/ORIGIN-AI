from pathlib import Path
import random

MAX_DATA_BYTES = 25 * 1024 * 1024

def load_texts(path: str | Path) -> list[str]:
    source = Path(path)
    if not source.exists() or not source.is_file(): raise FileNotFoundError(f"Dataset file not found: {source}")
    if source.stat().st_size > MAX_DATA_BYTES: raise ValueError("Dataset exceeds the 25 MB safety limit.")
    texts = [line.strip() for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not texts: raise ValueError("Dataset contains no non-empty text lines.")
    return list(dict.fromkeys(texts))

def split_texts(texts: list[str], validation_fraction: float = 0.1, seed: int = 42):
    if not 0 <= validation_fraction < 1 or not texts: raise ValueError("Invalid split fraction or empty dataset.")
    values = list(texts); random.Random(seed).shuffle(values)
    n_val = min(len(values) - 1, int(len(values) * validation_fraction))
    return values[n_val:], values[:n_val]
