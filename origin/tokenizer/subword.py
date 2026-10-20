"""Trainable local SentencePiece BPE tokenizer; never downloads model assets."""
from __future__ import annotations

import base64
import json
from pathlib import Path
import tempfile

import sentencepiece as spm


class SubwordTokenizer:
    def __init__(self, model_proto: bytes):
        self._processor = spm.SentencePieceProcessor(model_proto=model_proto)
        self._model_proto = model_proto
        self.vocab_size = self._processor.vocab_size()
        self.eos_id = self._processor.eos_id()
        self.pad_id = self._processor.pad_id()

    @property
    def model_proto(self) -> bytes:
        return self._model_proto

    @classmethod
    def train(cls, corpus_path: str | Path, vocab_size: int = 2048, character_coverage: float = 0.9995) -> "SubwordTokenizer":
        corpus = Path(corpus_path)
        if not corpus.is_file() or corpus.stat().st_size == 0:
            raise ValueError("Tokenizer training corpus is missing or empty.")
        if vocab_size < 256 or not 0.98 <= character_coverage <= 1.0:
            raise ValueError("BPE vocab_size must be at least 256 and character coverage in [0.98, 1.0].")
        with tempfile.TemporaryDirectory(prefix="origin-tokenizer-") as temp:
            prefix = Path(temp) / "origin"
            spm.SentencePieceTrainer.train(
                input=str(corpus), model_prefix=str(prefix), vocab_size=vocab_size,
                model_type="bpe", character_coverage=character_coverage,
                pad_id=0, unk_id=1, bos_id=2, eos_id=3,
                pad_piece="[PAD]", unk_piece="[UNK]", bos_piece="[BOS]", eos_piece="[EOS]",
                hard_vocab_limit=False, shuffle_input_sentence=False,
                input_sentence_size=0, num_threads=1, minloglevel=2,
            )
            return cls((prefix.with_suffix(".model")).read_bytes())

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = self._processor.encode(text, out_type=int)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        return self._processor.decode([int(token) for token in ids if int(token) not in {self.pad_id, self.eos_id}])

    def save(self, path: str | Path) -> None:
        payload = {"type": "sentencepiece_bpe", "version": 1, "model_proto": base64.b64encode(self._model_proto).decode("ascii")}
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SubwordTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("type") != "sentencepiece_bpe" or payload.get("version") != 1:
            raise ValueError("Invalid SentencePiece tokenizer metadata.")
        try:
            proto = base64.b64decode(payload["model_proto"], validate=True)
        except (KeyError, ValueError) as exc:
            raise ValueError("Invalid SentencePiece tokenizer model.") from exc
        return cls(proto)
