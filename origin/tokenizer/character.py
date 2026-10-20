"""Small deterministic character tokenizer; vocabulary is saved with checkpoints."""
from __future__ import annotations

import json
from pathlib import Path


class CharacterTokenizer:
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, vocabulary: list[str] | None = None):
        self.vocabulary = list(vocabulary or [self.PAD, self.UNK])
        if len(set(self.vocabulary)) != len(self.vocabulary) or self.PAD not in self.vocabulary or self.UNK not in self.vocabulary:
            raise ValueError("Vocabulary must contain unique entries including <pad> and <unk>.")
        self._ids = {token: index for index, token in enumerate(self.vocabulary)}

    @classmethod
    def from_texts(cls, texts: list[str]) -> "CharacterTokenizer":
        chars = sorted({char for text in texts for char in text})
        return cls([cls.PAD, cls.UNK, *chars])

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    @property
    def pad_id(self) -> int:
        return self._ids[self.PAD]

    def encode(self, text: str) -> list[int]:
        return [self._ids.get(char, self._ids[self.UNK]) for char in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocabulary[i] if 0 <= i < len(self.vocabulary) else self.UNK for i in ids if i != self.pad_id and i != self._ids[self.UNK])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"type": "character", "vocabulary": self.vocabulary}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharacterTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("type") != "character" or not isinstance(data.get("vocabulary"), list):
            raise ValueError("Invalid tokenizer metadata.")
        return cls(data["vocabulary"])
