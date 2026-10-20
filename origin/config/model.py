from dataclasses import asdict, dataclass

@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 128
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.0

    def validate(self) -> None:
        if self.vocab_size < 2 or self.block_size < 2 or min(self.n_embd, self.n_head, self.n_layer) < 1:
            raise ValueError("Model dimensions must be positive and vocabulary/block size at least 2.")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)
