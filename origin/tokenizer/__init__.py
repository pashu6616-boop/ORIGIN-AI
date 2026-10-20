from .character import CharacterTokenizer
from .subword import SubwordTokenizer
import json
from pathlib import Path

def load_tokenizer(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("type") == "character":
        return CharacterTokenizer.load(path)
    if data.get("type") == "sentencepiece_bpe":
        return SubwordTokenizer.load(path)
    raise ValueError("Unsupported tokenizer metadata.")

__all__ = ["CharacterTokenizer", "SubwordTokenizer", "load_tokenizer"]
