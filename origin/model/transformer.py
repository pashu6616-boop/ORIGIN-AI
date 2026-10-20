"""Minimal decoder-only Transformer with explicit causal attention masking."""
import torch
from torch import nn
from torch.nn import functional as F
from origin.config import ModelConfig

class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("causal", torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)), persistent=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, length, self.heads, self.head_dim)
        q, k, v = (part.view(shape).transpose(1, 2) for part in (q, k, v))
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(~self.causal[:length, :length], torch.finfo(scores.dtype).min)
        weights = self.dropout(F.softmax(scores, dim=-1))
        out = (weights @ v).transpose(1, 2).contiguous().view(batch, length, width)
        return self.dropout(self.proj(out))

class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.n_embd)
        self.ff = nn.Sequential(nn.Linear(config.n_embd, 4 * config.n_embd), nn.GELU(), nn.Linear(4 * config.n_embd, config.n_embd), nn.Dropout(config.dropout))
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ff(self.norm2(x))

class DecoderTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.n_layer)))
        self.norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize)
    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        if tokens.ndim != 2 or tokens.size(1) > self.config.block_size:
            raise ValueError("tokens must have shape (batch, time) within block_size")
        positions = torch.arange(tokens.size(1), device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        logits = self.lm_head(self.norm(self.blocks(x)))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        return logits, loss
    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, max_new_tokens: int = 80, temperature: float = 1.0, top_k: int | None = None):
        if max_new_tokens < 0 or temperature <= 0 or (top_k is not None and top_k < 1):
            raise ValueError("Invalid generation settings.")
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(tokens[:, -self.config.block_size:])
            next_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < values[:, [-1]]] = -torch.inf
            tokens = torch.cat((tokens, torch.multinomial(F.softmax(next_logits, dim=-1), 1)), dim=1)
        return tokens
